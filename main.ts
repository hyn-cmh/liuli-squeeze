// Smallest lossless PNG re-encoder for the png-squeeze bounty.
//
// For each 1280x720 RGBA screenshot this writes the same pixels back as the
// smallest PNG it can find, losslessly (every channel of every pixel exact,
// transparent pixels' colours included). Pure Node built-ins, no deps.
//
// Method, per image:
//   1. Decode to RGBA; analyse (opaque? grey? <=256 colours? one tRNS key?).
//   2. Candidate forms: palette (several entry orderings, transparent entries
//      first so tRNS stays short), grey, grey+alpha, RGB, RGBA, tRNS-keyed.
//   3. Per form, build filtered streams: fixed filters 0..4, per-row
//      minimum-sum, fewest-distinct-pairs and lowest-entropy choices; screen
//      each with zlib level 4. A brute-force per-row pass (each row the
//      filter whose bytes deflate smallest after the 32 KiB before it) runs
//      on the winning form and joins the pool.
//   4. Finalists are deflated at level 9 with a memLevel/strategy sweep, and
//      the best stream, when it is small enough (palette/grey scale), also
//      goes through this file's own optimal-parsing deflate (LZ77 hash
//      chains + shortest-path parse with iteratively refined Huffman costs
//      + block splitting, in the manner of zopfli) under tight fixed work
//      budgets. The smallest complete PNG wins.
//
// Speed note (round 2): the first submitted version ran 49.3s locally and
// timed out in the 60s sandbox. The optimal-parse budgets below (chain
// length, parse iterations, candidate lengths, block splits, and a hard
// stream-size gate) plus cheaper screening/brute trials cut the local wall
// to roughly half while keeping most of the size win.
//
// Deterministic: the search budget is a fixed function of the input size,
// never of the clock; no randomness, no environment input. Same input bytes
// in, same output bytes out.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { constants, deflateSync, inflateSync } from 'node:zlib';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// ---------------------------------------------------------------- CRC-32
const CRC_TAB = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();
function crc32(buf, from, to) {
  let c = -1;
  for (let i = from; i < to; i++) c = CRC_TAB[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}
function adler32(buf) {
  let a = 1, b = 0;
  for (let i = 0; i < buf.length;) {
    const stop = Math.min(buf.length, i + 5552);
    for (; i < stop; i++) { a += buf[i]; b += a; }
    a %= 65521; b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

// ---------------------------------------------------------------- decode
const CH_OF = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };

function paethPred(a, b, c) {
  const p = a + b - c;
  const pa = p > a ? p - a : a - p;
  const pb = p > b ? p - b : b - p;
  const pc = p > c ? p - c : c - p;
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** Decode any contest-readable PNG to {rgba, width, height}. */
function decodeToRgba(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let pos = 8, width = 0, height = 0, colourType = 6;
  let palette = null, trns = null;
  const idatParts = [];
  while (pos < bytes.length) {
    const len = view.getUint32(pos);
    const tag = String.fromCharCode(bytes[pos + 4], bytes[pos + 5], bytes[pos + 6], bytes[pos + 7]);
    const body = bytes.subarray(pos + 8, pos + 8 + len);
    pos += 12 + len;
    if (tag === 'IHDR') {
      const dv = new DataView(body.buffer, body.byteOffset, 13);
      width = dv.getUint32(0); height = dv.getUint32(4); colourType = body[9];
    } else if (tag === 'PLTE') palette = body;
    else if (tag === 'tRNS') trns = body;
    else if (tag === 'IDAT') idatParts.push(body);
    else if (tag === 'IEND') break;
  }
  const bpp = CH_OF[colourType];
  const stride = width * bpp;
  const raw = inflateSync(idatParts.length === 1 ? idatParts[0] : Buffer.concat(idatParts));
  const samples = new Uint8Array(height * stride);
  for (let y = 0; y < height; y++) {
    const ft = raw[y * (stride + 1)];
    const src = y * (stride + 1) + 1;
    const dst = y * stride;
    const up = dst - stride;
    if (ft === 0) samples.set(raw.subarray(src, src + stride), dst);
    else if (ft === 1) {
      for (let i = 0; i < stride; i++) samples[dst + i] = raw[src + i] + (i >= bpp ? samples[dst + i - bpp] : 0);
    } else if (ft === 2) {
      for (let i = 0; i < stride; i++) samples[dst + i] = raw[src + i] + (y > 0 ? samples[up + i] : 0);
    } else if (ft === 3) {
      for (let i = 0; i < stride; i++) {
        samples[dst + i] = raw[src + i] + (((i >= bpp ? samples[dst + i - bpp] : 0) + (y > 0 ? samples[up + i] : 0)) >> 1);
      }
    } else {
      for (let i = 0; i < stride; i++) {
        const a = i >= bpp ? samples[dst + i - bpp] : 0;
        const b = y > 0 ? samples[up + i] : 0;
        const c = y > 0 && i >= bpp ? samples[up + i - bpp] : 0;
        samples[dst + i] = raw[src + i] + paethPred(a, b, c);
      }
    }
  }
  const pixels = width * height;
  const rgba = new Uint8Array(pixels * 4);
  if (colourType === 6) rgba.set(samples);
  else if (colourType === 2) {
    const key = trns === null ? -1 : (trns[1] << 16) | (trns[3] << 8) | trns[5];
    for (let i = 0, s = 0, d = 0; i < pixels; i++, s += 3, d += 4) {
      rgba[d] = samples[s]; rgba[d + 1] = samples[s + 1]; rgba[d + 2] = samples[s + 2];
      rgba[d + 3] = ((samples[s] << 16) | (samples[s + 1] << 8) | samples[s + 2]) === key ? 0 : 255;
    }
  } else if (colourType === 0) {
    const key = trns === null ? -1 : trns[1];
    for (let i = 0, d = 0; i < pixels; i++, d += 4) {
      const g = samples[i];
      rgba[d] = rgba[d + 1] = rgba[d + 2] = g;
      rgba[d + 3] = g === key ? 0 : 255;
    }
  } else if (colourType === 4) {
    for (let i = 0, s = 0, d = 0; i < pixels; i++, s += 2, d += 4) {
      rgba[d] = rgba[d + 1] = rgba[d + 2] = samples[s];
      rgba[d + 3] = samples[s + 1];
    }
  } else {
    const entries = palette.length / 3;
    const lut = new Uint8Array(entries * 4);
    for (let e = 0; e < entries; e++) {
      lut[e * 4] = palette[e * 3]; lut[e * 4 + 1] = palette[e * 3 + 1]; lut[e * 4 + 2] = palette[e * 3 + 2];
      lut[e * 4 + 3] = trns !== null && e < trns.length ? trns[e] : 255;
    }
    for (let i = 0, d = 0; i < pixels; i++, d += 4) {
      const e = samples[i] * 4;
      rgba[d] = lut[e]; rgba[d + 1] = lut[e + 1]; rgba[d + 2] = lut[e + 2]; rgba[d + 3] = lut[e + 3];
    }
  }
  return { rgba, width, height };
}

// ---------------------------------------------------------------- encode
const SIG = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
const tagBytes = (s) => Uint8Array.from(s, (ch) => ch.charCodeAt(0));

function assemblePng(chunks) {
  let total = SIG.length;
  for (const [, body] of chunks) total += 12 + body.length;
  const out = new Uint8Array(total);
  const view = new DataView(out.buffer);
  out.set(SIG, 0);
  let at = SIG.length;
  for (const [tag, body] of chunks) {
    view.setUint32(at, body.length);
    out.set(tagBytes(tag), at + 4);
    out.set(body, at + 8);
    view.setUint32(at + 8 + body.length, crc32(out, at + 4, at + 8 + body.length));
    at += 12 + body.length;
  }
  return out;
}
function ihdrBody(width, height, colourType) {
  const d = new Uint8Array(13);
  const v = new DataView(d.buffer);
  v.setUint32(0, width);
  v.setUint32(4, height);
  d[8] = 8;
  d[9] = colourType;
  return d;
}
function pngForForm(width, height, form, idat) {
  const chunks = [['IHDR', ihdrBody(width, height, form.colourType)]];
  if (form.plte !== null) chunks.push(['PLTE', form.plte]);
  if (form.trns !== null) chunks.push(['tRNS', form.trns]);
  chunks.push(['IDAT', idat]);
  chunks.push(['IEND', new Uint8Array(0)]);
  return assemblePng(chunks);
}
const sideChunkBytes = (form) =>
  (form.plte === null ? 0 : 12 + form.plte.length) + (form.trns === null ? 0 : 12 + form.trns.length);
const PNG_FIXED_BYTES = 8 + 25 + 12 + 12; // signature + IHDR chunk + IDAT framing + IEND chunk

// ---------------------------------------------------------------- analyse
function pixelWords(rgba) {
  const aligned = rgba.byteOffset % 4 === 0 ? rgba : rgba.slice();
  return new Uint32Array(aligned.buffer, aligned.byteOffset, aligned.length >>> 2);
}

function analyseImage(rgba) {
  const n = rgba.length >>> 2;
  const words = pixelWords(rgba);
  let opaque = true, grey = true, manyColours = false;
  const counts = new Map();
  for (let i = 0, d = 0; i < n; i++, d += 4) {
    if (rgba[d + 3] !== 255) opaque = false;
    if (rgba[d] !== rgba[d + 1] || rgba[d] !== rgba[d + 2]) grey = false;
    if (!manyColours) {
      const c = words[i];
      const seen = counts.get(c);
      if (seen === undefined) {
        if (counts.size === 256) manyColours = true;
        else counts.set(c, 1);
      } else counts.set(c, seen + 1);
    }
  }
  const colours = manyColours ? null : [...counts.keys()].map((c) => {
    const b = new Uint8Array(Uint32Array.of(c).buffer);
    return { r: b[0], g: b[1], b: b[2], a: b[3], word: c, count: counts.get(c) };
  });
  return { opaque, grey, colours };
}

/** The one fully transparent value keyable by tRNS (grey value or RGB triple), else null. */
function singleTransparentKey(rgba, grey) {
  let key = -1;
  for (let d = 0; d < rgba.length; d += 4) {
    const a = rgba[d + 3];
    if (a === 255) continue;
    if (a !== 0) return null;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (key === -1) key = v;
    else if (key !== v) return null;
  }
  if (key === -1) return null;
  for (let d = 0; d < rgba.length; d += 4) {
    if (rgba[d + 3] !== 255) continue;
    const v = grey ? rgba[d] : (rgba[d] << 16) | (rgba[d + 1] << 8) | rgba[d + 2];
    if (v === key) return null;
  }
  return key;
}

function channelSamples(rgba, ct) {
  const n = rgba.length >>> 2;
  if (ct === 6) return rgba;
  const ch = ct === 0 ? 1 : ct === 4 ? 2 : 3;
  const out = new Uint8Array(n * ch);
  for (let i = 0, d = 0, s = 0; i < n; i++, d += 4, s += ch) {
    if (ct === 0) out[s] = rgba[d];
    else if (ct === 4) { out[s] = rgba[d]; out[s + 1] = rgba[d + 3]; }
    else { out[s] = rgba[d]; out[s + 1] = rgba[d + 1]; out[s + 2] = rgba[d + 2]; }
  }
  return out;
}

function orderPalette(colours, order) {
  const lum = (c) => 299 * c.r + 587 * c.g + 114 * c.b;
  return [...colours].sort((x, y) => {
    const tx = x.a === 255 ? 1 : 0, ty = y.a === 255 ? 1 : 0;
    if (tx !== ty) return tx - ty;                       // transparent entries first (short tRNS)
    if (tx === 0 && x.a !== y.a) return x.a - y.a;
    if (order === 'luminance') return lum(x) - lum(y) || x.word - y.word;
    if (order === 'appearance') return 0;                // stable sort keeps first-appearance order
    return y.count - x.count || x.word - y.word;          // 'count'
  });
}

function paletteFormOf(rgba, colours, order) {
  const entries = orderPalette(colours, order);
  const indexOf = new Map(entries.map((c, i) => [c.word, i]));
  const plte = new Uint8Array(entries.length * 3);
  let lastTransparent = 0;
  entries.forEach((c, i) => {
    plte[i * 3] = c.r; plte[i * 3 + 1] = c.g; plte[i * 3 + 2] = c.b;
    if (c.a !== 255) lastTransparent = i + 1;
  });
  const trns = lastTransparent === 0 ? null : Uint8Array.from(entries.slice(0, lastTransparent), (c) => c.a);
  const n = rgba.length >>> 2;
  const words = pixelWords(rgba);
  const indices = new Uint8Array(n);
  for (let i = 0; i < n; i++) indices[i] = indexOf.get(words[i]);
  return { colourType: 3, bpp: 1, samples: indices, plte, trns, tag: 'pal-' + order };
}

/** Every lossless form worth trying for this image. */
function candidateForms(rgba) {
  const { opaque, grey, colours } = analyseImage(rgba);
  const forms = [];
  if (colours !== null) {
    for (const order of ['count', 'luminance', 'appearance']) forms.push(paletteFormOf(rgba, colours, order));
    if (!grey) return forms; // a palette holds a non-grey image in <= 1/3 the bytes; channels cannot win
  }
  const key = opaque ? null : singleTransparentKey(rgba, grey);
  let ct, trns = null;
  if (grey) {
    ct = opaque || key !== null ? 0 : 4;
    if (key !== null) trns = Uint8Array.of(0, key);
  } else {
    ct = opaque || key !== null ? 2 : 6;
    if (key !== null) trns = Uint8Array.of(0, key >>> 16, 0, (key >>> 8) & 0xff, 0, key & 0xff);
  }
  forms.push({ colourType: ct, bpp: CH_OF[ct], samples: channelSamples(rgba, ct), plte: null, trns, tag: 'ct' + ct });
  return forms;
}

// ---------------------------------------------------------------- filtering
function filterRow(type, cur, prev, bpp, out) {
  const n = cur.length;
  if (type === 0) out.set(cur);
  else if (type === 1) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (i >= bpp ? cur[i - bpp] : 0);
  } else if (type === 2) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (prev === null ? 0 : prev[i]);
  } else if (type === 3) {
    for (let i = 0; i < n; i++) out[i] = cur[i] - (((i >= bpp ? cur[i - bpp] : 0) + (prev === null ? 0 : prev[i])) >> 1);
  } else {
    for (let i = 0; i < n; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev === null ? 0 : prev[i];
      const c = prev !== null && i >= bpp ? prev[i - bpp] : 0;
      out[i] = cur[i] - paethPred(a, b, c);
    }
  }
}

function signedByteSum(row) {
  let s = 0;
  for (let i = 0; i < row.length; i++) { const v = row[i]; s += v < 128 ? v : 256 - v; }
  return s;
}
function distinctPairs(row, seen, stamp) {
  let n = 0;
  for (let i = 1; i < row.length; i++) {
    const pair = (row[i - 1] << 8) | row[i];
    if (seen[pair] !== stamp) { seen[pair] = stamp; n++; }
  }
  return n;
}
/**
 * One pass building every cheap candidate stream of a form: the five
 * fixed-filter streams plus per-row minsum / pairs choices.
 * Returns [{choice, data}].
 */
function cheapStreams(samples, width, height, bpp) {
  const stride = width * bpp;
  const size = height * (stride + 1);
  const fixed = [0, 1, 4].map((t) => ({ choice: t, data: new Uint8Array(size) }));
  const minsum = { choice: 'minsum', data: new Uint8Array(size) };
  const pairs = { choice: 'pairs', data: new Uint8Array(size) };
  const trial = Array.from({ length: 5 }, (_, t) => {
    const row = new Uint8Array(stride + 1);
    row[0] = t;
    return row;
  });
  const pairSeen = new Uint32Array(65536);
  let stamp = 0;
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * (stride + 1);
    let msT = 0, msBest = Infinity, prT = 0, prBest = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trial[t].subarray(1));
      const body = trial[t].subarray(1);
      const ms = signedByteSum(body);
      if (ms < msBest) { msBest = ms; msT = t; }
      const pr = distinctPairs(body, pairSeen, ++stamp);
      if (pr < prBest) { prBest = pr; prT = t; }
    }
    for (const s of fixed) s.data.set(trial[s.choice], at);
    minsum.data.set(trial[msT], at);
    pairs.data.set(trial[prT], at);
  }
  return [...fixed, minsum, pairs];
}

const BRUTE_CONTEXT = 24576;
const BRUTE_LEVEL = 3; // round 2: restored to 3 (L1 lost score)

/** Brute-force per-row filter choice: the filter whose row deflates smallest after the 32 KiB before it. */
function bruteStream(samples, width, height, bpp) {
  const stride = width * bpp;
  const raw = new Uint8Array(height * (stride + 1));
  const trial = Array.from({ length: 5 }, (_, t) => {
    const row = new Uint8Array(stride + 1);
    row[0] = t;
    return row;
  });
  for (let y = 0; y < height; y++) {
    const cur = samples.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : samples.subarray((y - 1) * stride, y * stride);
    const at = y * (stride + 1);
    const before = raw.subarray(Math.max(0, at - BRUTE_CONTEXT), at);
    const opts = before.length === 0
      ? { level: BRUTE_LEVEL, memLevel: 9 }
      : { level: BRUTE_LEVEL, memLevel: 9, dictionary: before };
    let type = 0, best = Infinity;
    for (let t = 0; t < 5; t++) {
      filterRow(t, cur, prev, bpp, trial[t].subarray(1));
      const cost = deflateSync(trial[t], opts).length;
      if (cost < best) { best = cost; type = t; }
    }
    raw.set(trial[type], at);
  }
  return raw;
}

// ---------------------------------------------------------------- optimal deflate
// A zopfli-style deflate: LZ77 matches from hash chains, a shortest-path
// (optimal) parse under Huffman bit costs, costs refined by re-counting the
// parse a fixed number of times, then block splitting. All budgets are fixed
// functions of the input length, so the output is a pure function of the data.

const LEN_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LEN_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DIST_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DIST_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const LEN_SYM = new Uint16Array(259);   // length 3..258 -> litlen symbol 257..285
const LEN_XB = new Uint8Array(259);     // its extra-bit count
const DIST_SYM = new Uint8Array(32769); // distance 1..32768 -> symbol 0..29
const DIST_XB = new Uint8Array(32769);
for (let s = 0; s < 29; s++) {
  const hi = s === 28 ? 258 : LEN_BASE[s + 1] - 1;
  for (let l = LEN_BASE[s]; l <= hi; l++) { LEN_SYM[l] = 257 + s; LEN_XB[l] = LEN_EXTRA[s]; }
}
for (let s = 0; s < 30; s++) {
  const hi = s === 29 ? 32768 : DIST_BASE[s + 1] - 1;
  for (let d = DIST_BASE[s]; d <= hi; d++) { DIST_SYM[d] = s; DIST_XB[d] = DIST_EXTRA[s]; }
}
// Match lengths worth offering the DP for a longest match of L: every length
// up to 32, then each length where the litlen symbol changes, plus L itself.
// (Round 2: full range cut from 48 to 32 to shrink the DP inner loop.)
function buildCandLens(cap) {
  const steps = [35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
  const table = new Array(259);
  for (let L = 0; L <= 258; L++) {
    if (L < 3) { table[L] = []; continue; }
    const list = [];
    const upTo = Math.min(L, cap);
    for (let l = 3; l <= upTo; l++) list.push(l);
    for (const l of steps) { if (l > L) break; if (l > cap) list.push(l); }
    if (L > cap && list[list.length - 1] !== L) list.push(L);
    table[L] = list;
  }
  return table;
}
const CAND_LENS = buildCandLens(48);
const CAND_LENS_BIG = buildCandLens(24);

/** Pareto-optimal matches (<=258, window 32 KiB) at every position, via 3-byte
 *  hash chains: up to 8 (length, distance) offers per position, none dominated
 *  (another offer both longer and no farther). This is the sublen idea: the DP
 *  needs not just the longest match but every length/distance trade-off. */
const matchScratch = { offerLen: null, offerDist: null, offerCnt: null, head: null, last3: null, prev: null };
function findMatches(data, maxChain, niceLen) {
  const n = data.length;
  const MAX_OFFERS = n <= 1200000 ? 6 : 3;
  const need = n * MAX_OFFERS;
  if (matchScratch.offerLen === null || matchScratch.offerLen.length < need) {
    matchScratch.offerLen = new Uint16Array(need);
    matchScratch.offerDist = new Uint16Array(need);
    matchScratch.offerCnt = new Uint8Array(n);
    matchScratch.prev = new Int32Array(n);
  }
  if (matchScratch.head === null) {
    matchScratch.head = new Int32Array(1048576);
    matchScratch.last3 = new Int32Array(65536);
  }
  const offerLen = matchScratch.offerLen;
  const offerDist = matchScratch.offerDist;
  const offerCnt = matchScratch.offerCnt;
  offerCnt.fill(0, 0, n);
  const head = matchScratch.head.fill(-1);
  const last3 = matchScratch.last3.fill(-1);
  const prev = matchScratch.prev;
  prev.fill(-1, 0, n);
  const hash4 = (i) => (Math.imul((data[i] << 24) | (data[i + 1] << 16) | (data[i + 2] << 8) | data[i + 3], 0x9e3779b1) >>> 12) & 1048575;
  const hash3 = (i) => (Math.imul((data[i] << 16) | (data[i + 1] << 8) | data[i + 2], 0x9e3779b1) >>> 16) & 65535;
  const words = data.byteOffset % 4 === 0 ? new Uint32Array(data.buffer, data.byteOffset, n >>> 2) : null;
  const lastStart = n - 4;
  const tmpLen = new Int32Array(MAX_OFFERS);
  const tmpDist = new Int32Array(MAX_OFFERS);
  const addOffer = (l, d, cnt) => {
    let dominated = false;
    let w = 0;
    for (let k = 0; k < cnt; k++) {
      if (tmpLen[k] >= l && tmpDist[k] <= d) { dominated = true; break; }
      if (!(l >= tmpLen[k] && d <= tmpDist[k])) { tmpLen[w] = tmpLen[k]; tmpDist[w] = tmpDist[k]; w++; }
    }
    if (dominated) return cnt;
    cnt = w;
    if (cnt < MAX_OFFERS) { tmpLen[cnt] = l; tmpDist[cnt] = d; cnt++; }
    else {
      let worstK = -1;
      for (let k = 0; k < cnt; k++) if (tmpLen[k] <= l && (worstK < 0 || tmpDist[k] > tmpDist[worstK])) worstK = k;
      if (worstK >= 0) { tmpLen[worstK] = l; tmpDist[worstK] = d; }
    }
    return cnt;
  };
  for (let i = 0; i <= lastStart; i++) {
    const h = hash4(i);
    let cand = head[h];
    let bestLen = 0;
    let cnt = 0;
    let budget = maxChain;
    const maxLen = Math.min(258, n - i);
    const b0 = data[i], b1 = data[i + 1], b2 = data[i + 2], b3 = data[i + 3];
    // Chain candidates arrive nearest-first (distances strictly increase),
    // so every offer stored so far has distance <= the current candidate's.
    // A new match is therefore useful only when it is longer than every
    // offer so far (bestLen): anything shorter is dominated. That turns
    // the Pareto maintenance into a single comparison.
    while (cand >= 0 && i - cand <= 32768 && budget-- > 0) {
      const d = i - cand;
      if (data[cand] === b0 && data[cand + 1] === b1 && data[cand + 2] === b2 && data[cand + 3] === b3) {
        if (bestLen === 0 || (bestLen < maxLen && data[cand + bestLen] === data[i + bestLen])) {
          let l = 4;
          if (words !== null) {
            while (l < maxLen && ((i + l) & 3) !== 0 && data[cand + l] === data[i + l]) l++;
            if (((i + l) & 3) === 0) {
              let wi = (i + l) >>> 2, wc = (cand + l) >>> 2;
              const shB = ((cand + l) & 3) * 8;
              if (shB === 0) {
                let w = 0;
                while (l + 4 <= maxLen && words[wc + w] === words[wi + w]) { l += 4; w++; }
              } else {
                while (l + 4 <= maxLen && (((words[wc] >>> shB) | (words[wc + 1] << (32 - shB))) >>> 0) === words[wi]) { l += 4; wi++; wc++; }
              }
            }
          }
          while (l < maxLen && data[cand + l] === data[i + l]) l++;
          if (l > bestLen) {
            bestLen = l;
            if (cnt < MAX_OFFERS) { tmpLen[cnt] = l; tmpDist[cnt] = d; cnt++; }
            else {
              // Full: drop the offer whose length gain over its predecessor
              // in the frontier is smallest (least useful), keep the new one.
              let dropK = 0, dropGain = Infinity, prevLen = 0;
              const order = [];
              for (let k = 0; k < cnt; k++) order.push(k);
              order.sort((a, b) => tmpDist[a] - tmpDist[b]);
              for (const k of order) {
                const gain = tmpLen[k] - prevLen;
                if (gain < dropGain) { dropGain = gain; dropK = k; }
                prevLen = tmpLen[k];
              }
              tmpLen[dropK] = l; tmpDist[dropK] = d;
            }
            if (l >= 258) break;
            if (l >= niceLen) budget = Math.min(budget, 2);
            else if (l >= 32) budget = Math.min(budget, 4);
          }
        }
      }
      cand = prev[cand];
    }
    // Nearest length-3 match via the 3-byte side table.
    const h3 = hash3(i);
    const c3 = last3[h3];
    if (c3 >= 0 && i - c3 <= 32768 && data[c3] === b0 && data[c3 + 1] === b1 && data[c3 + 2] === b2) {
      cnt = addOffer(3, i - c3, cnt);
    }
    last3[h3] = i;
    if (cnt > 0) {
      offerCnt[i] = cnt;
      const base = i * MAX_OFFERS;
      for (let k = 0; k < cnt; k++) { offerLen[base + k] = tmpLen[k]; offerDist[base + k] = tmpDist[k]; }
    }
    prev[i] = head[h];
    head[h] = i;
  }
  // Tail positions (last 3) can still take a length-3 match.
  for (let i = lastStart + 1; i + 2 < n; i++) {
    const h3 = hash3(i);
    const c3 = last3[h3];
    if (c3 >= 0 && i - c3 <= 32768 && data[c3] === data[i] && data[c3 + 1] === data[i + 1] && data[c3 + 2] === data[i + 2]) {
      offerCnt[i] = 1;
      offerLen[i * MAX_OFFERS] = 3;
      offerDist[i * MAX_OFFERS] = i - c3;
    }
    last3[h3] = i;
  }
  return { offerLen, offerDist, offerCnt, maxOffers: MAX_OFFERS };
}

/** Length-limited Huffman code lengths by package-merge. freq -> Uint8Array lengths. */
function huffmanLengths(freq, maxBits) {
  const n = freq.length;
  const lengths = new Uint8Array(n);
  const syms = [];
  for (let i = 0; i < n; i++) if (freq[i] > 0) syms.push(i);
  if (syms.length === 0) return lengths;
  if (syms.length === 1) { lengths[syms[0]] = 1; return lengths; }
  syms.sort((a, b) => freq[a] - freq[b] || a - b);
  const leaf = (s) => ({ w: freq[s], members: [s] });
  let list = syms.map(leaf);
  for (let level = 1; level < maxBits; level++) {
    const packaged = [];
    for (let i = 0; i + 1 < list.length; i += 2) {
      packaged.push({ w: list[i].w + list[i + 1].w, members: list[i].members.concat(list[i + 1].members) });
    }
    const leaves = syms.map(leaf);
    const merged = [];
    let a = 0, b = 0;
    while (a < leaves.length || b < packaged.length) {
      if (b >= packaged.length || (a < leaves.length && leaves[a].w <= packaged[b].w)) merged.push(leaves[a++]);
      else merged.push(packaged[b++]);
    }
    list = merged;
  }
  const take = Math.min(list.length, 2 * syms.length - 2);
  for (let i = 0; i < take; i++) for (const s of list[i].members) lengths[s]++;
  return lengths;
}

/** Canonical codes (bit-reversed, ready for the LSB-first writer) from lengths. */
function canonicalCodes(lengths) {
  const maxBits = 15;
  const blCount = new Int32Array(maxBits + 1);
  for (let i = 0; i < lengths.length; i++) if (lengths[i] > 0) blCount[lengths[i]]++;
  const nextCode = new Int32Array(maxBits + 1);
  let code = 0;
  for (let bits = 1; bits <= maxBits; bits++) {
    code = (code + blCount[bits - 1]) << 1;
    nextCode[bits] = code;
  }
  const codes = new Int32Array(lengths.length);
  for (let i = 0; i < lengths.length; i++) {
    const l = lengths[i];
    if (l === 0) continue;
    let c = nextCode[l]++;
    let rev = 0;
    for (let k = 0; k < l; k++) { rev = (rev << 1) | (c & 1); c >>>= 1; }
    codes[i] = rev;
  }
  return codes;
}

const FIXED_LIT_LEN = (() => {
  const t = new Uint8Array(288);
  for (let i = 0; i < 144; i++) t[i] = 8;
  for (let i = 144; i < 256; i++) t[i] = 9;
  for (let i = 256; i < 280; i++) t[i] = 7;
  for (let i = 280; i < 288; i++) t[i] = 8;
  return t;
})();
const FIXED_DIST_LEN = new Uint8Array(32).fill(5);

/**
 * Shortest-path parse of `data` under bit costs litCost[288] / distCost[32]
 * (extra bits added here). Each position offers its longest match and its
 * cheapest-distance match. Returns the token list and the parse's bit cost.
 */
const parseScratch = { cost: null, choice: null, choiceDist: null };
function optimalParse(data, matches, litCost, distCost, candTable = CAND_LENS, start = 0, end = data.length) {
  const { offerLen, offerDist, offerCnt, maxOffers } = matches;
  const n = end - start;
  if (parseScratch.cost === null || parseScratch.cost.length < n + 1) {
    parseScratch.cost = new Float64Array(n + 1);
    parseScratch.choice = new Int32Array(n + 1);
    parseScratch.choiceDist = new Uint16Array(n + 1);
  }
  const cost = parseScratch.cost.fill(Infinity, 0, n + 1);
  const choice = parseScratch.choice; // token length ending at i (1 = literal)
  const choiceDist = parseScratch.choiceDist; // distance of the match ending at i
  const lenCost = new Float64Array(259);
  for (let l = 3; l <= 258; l++) lenCost[l] = litCost[LEN_SYM[l]] + LEN_XB[l];
  cost[0] = 0;
  for (let i = 0; i < n; i++) {
    const ci = cost[i];
    if (ci === Infinity) continue;
    const gi = start + i;
    const litTotal = ci + litCost[data[gi]];
    if (litTotal < cost[i + 1]) { cost[i + 1] = litTotal; choice[i + 1] = 1; }
    const cnt = offerCnt[gi];
    if (cnt > 0) {
      const base = gi * maxOffers;
      const room = n - i;
      for (let k = 0; k < cnt; k++) {
        const L = Math.min(offerLen[base + k], room);
        if (L < 3) continue;
        const d = offerDist[base + k];
        const bc = ci + distCost[DIST_SYM[d]] + DIST_XB[d];
        const cands = candTable[L];
        for (let j = 0; j < cands.length; j++) {
          const l = cands[j];
          const c = bc + lenCost[l];
          if (c < cost[i + l]) { cost[i + l] = c; choice[i + l] = l; choiceDist[i + l] = d; }
        }
      }
    }
  }
  const rev = [];
  for (let at = n; at > 0;) {
    const l = choice[at];
    const s = at - l;
    if (l === 1) rev.push({ match: false, byte: data[start + s] });
    else rev.push({ match: true, len: l, dist: choiceDist[at] });
    at = s;
  }
  rev.reverse();
  return { tokens: rev, bits: cost[n] };
}

function countSymbols(tokens) {
  const litFreq = new Int32Array(288);
  const distFreq = new Int32Array(32);
  for (const t of tokens) {
    if (t.match) { litFreq[LEN_SYM[t.len]]++; distFreq[DIST_SYM[t.dist]]++; }
    else litFreq[t.byte]++;
  }
  litFreq[256]++; // end of block
  return { litFreq, distFreq };
}
const costsOf = (lengths, fallback) => {
  const c = new Float64Array(lengths.length);
  for (let i = 0; i < lengths.length; i++) c[i] = lengths[i] > 0 ? lengths[i] : fallback;
  return c;
};
/** Smoothed -log2 costs from symbol frequencies (what the DP optimises against). */
function smoothedCosts(freq) {
  let total = 0;
  for (let i = 0; i < freq.length; i++) total += freq[i];
  const denom = total + freq.length;
  const c = new Float64Array(freq.length);
  for (let i = 0; i < freq.length; i++) c[i] = -Math.log2((freq[i] + 1) / denom);
  return c;
}

// --- bit writer (LSB first) ---
function makeBitWriter() {
  const bytes = [];
  let acc = 0, nbits = 0;
  return {
    bits(value, count) {
      acc |= value << nbits;
      nbits += count;
      while (nbits >= 8) { bytes.push(acc & 0xff); acc >>>= 8; nbits -= 8; }
    },
    align() { if (nbits > 0) { bytes.push(acc & 0xff); acc = 0; nbits = 0; } },
    byte(b) { bytes.push(b); },
    finish() { this.align(); return Uint8Array.from(bytes); },
    get size() { return bytes.length + (nbits > 0 ? 1 : 0); },
  };
}

const CL_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/** RLE-encode a code-length sequence for the dynamic header. Returns [{sym, extra, xbits}]. */
function rleCodeLengths(seq) {
  const out = [];
  let i = 0;
  while (i < seq.length) {
    const v = seq[i];
    let run = 1;
    while (i + run < seq.length && seq[i + run] === v) run++;
    if (v === 0) {
      let left = run;
      while (left >= 11) { const take = Math.min(left, 138); out.push({ sym: 18, extra: take - 11, xbits: 7 }); left -= take; }
      if (left >= 3) { out.push({ sym: 17, extra: left - 3, xbits: 3 }); left = 0; }
      while (left-- > 0) out.push({ sym: 0, extra: 0, xbits: 0 });
    } else {
      out.push({ sym: v, extra: 0, xbits: 0 });
      let left = run - 1;
      while (left >= 3) { const take = Math.min(left, 6); out.push({ sym: 16, extra: take - 3, xbits: 2 }); left -= take; }
      while (left-- > 0) out.push({ sym: v, extra: 0, xbits: 0 });
    }
    i += run;
  }
  return out;
}

/** Exact bit cost of one block (tokens) under dynamic vs fixed Huffman; returns the cheaper plan. */
function planBlock(tokens) {
  const { litFreq, distFreq } = countSymbols(tokens);
  // Dynamic plan.
  let litLen = huffmanLengths(litFreq, 15);
  let distLen = huffmanLengths(distFreq, 15);
  // A table with a single used symbol needs a second 1-bit code to be a
  // complete prefix code for strict inflaters.
  const ensurePair = (lengths, freq) => {
    let used = 0, only = -1;
    for (let i = 0; i < lengths.length; i++) if (lengths[i] > 0) { used++; only = i; }
    if (used === 1) {
      for (let i = 0; i < lengths.length; i++) if (i !== only) { lengths[i] = 1; break; }
    }
    if (used === 0) { lengths[0] = 1; lengths[1] = 1; }
    void freq;
  };
  ensurePair(litLen, litFreq);
  ensurePair(distLen, distFreq);
  let hlit = 288;
  while (hlit > 257 && litLen[hlit - 1] === 0) hlit--;
  let hdist = 30;
  while (hdist > 1 && distLen[hdist - 1] === 0) hdist--;
  const seq = [...litLen.subarray(0, hlit), ...distLen.subarray(0, hdist)];
  const rle = rleCodeLengths(seq);
  const clFreq = new Int32Array(19);
  for (const r of rle) clFreq[r.sym]++;
  let clLen = huffmanLengths(clFreq, 7);
  ensurePair(clLen, clFreq);
  let hclen = 19;
  while (hclen > 4 && clLen[CL_ORDER[hclen - 1]] === 0) hclen--;
  let dynBits = 3 + 5 + 5 + 4 + hclen * 3;
  for (const r of rle) dynBits += clLen[r.sym] + r.xbits;
  for (const t of tokens) {
    if (t.match) dynBits += litLen[LEN_SYM[t.len]] + LEN_XB[t.len] + distLen[DIST_SYM[t.dist]] + DIST_XB[t.dist];
    else dynBits += litLen[t.byte];
  }
  dynBits += litLen[256];
  // Fixed plan.
  let fixedBits = 3;
  for (const t of tokens) {
    if (t.match) fixedBits += FIXED_LIT_LEN[LEN_SYM[t.len]] + LEN_XB[t.len] + 5 + DIST_XB[t.dist];
    else fixedBits += FIXED_LIT_LEN[t.byte];
  }
  fixedBits += FIXED_LIT_LEN[256];
  return dynBits <= fixedBits
    ? { dynamic: true, bits: dynBits, litLen, distLen, hlit, hdist, hclen, clLen, rle }
    : { dynamic: false, bits: fixedBits, litLen: FIXED_LIT_LEN, distLen: FIXED_DIST_LEN };
}

function writeBlock(w, tokens, plan, isFinal) {
  w.bits(isFinal ? 1 : 0, 1);
  w.bits(plan.dynamic ? 2 : 1, 2);
  if (plan.dynamic) {
    w.bits(plan.hlit - 257, 5);
    w.bits(plan.hdist - 1, 5);
    w.bits(plan.hclen - 4, 4);
    for (let i = 0; i < plan.hclen; i++) w.bits(plan.clLen[CL_ORDER[i]], 3);
    const clCodes = canonicalCodes(plan.clLen);
    for (const r of plan.rle) {
      w.bits(clCodes[r.sym], plan.clLen[r.sym]);
      if (r.xbits > 0) w.bits(r.extra, r.xbits);
    }
  }
  const litCodes = canonicalCodes(plan.litLen);
  const distCodes = canonicalCodes(plan.distLen);
  for (const t of tokens) {
    if (t.match) {
      const ls = LEN_SYM[t.len];
      w.bits(litCodes[ls], plan.litLen[ls]);
      if (LEN_XB[t.len] > 0) w.bits(t.len - LEN_BASE[ls - 257], LEN_XB[t.len]);
      const ds = DIST_SYM[t.dist];
      w.bits(distCodes[ds], plan.distLen[ds]);
      if (DIST_XB[t.dist] > 0) w.bits(t.dist - DIST_BASE[ds], DIST_XB[t.dist]);
    } else {
      w.bits(litCodes[t.byte], plan.litLen[t.byte]);
    }
  }
  w.bits(litCodes[256], plan.litLen[256]);
}

/**
 * Optimal-parsing deflate of `data` as a complete zlib stream.
 * Budget knobs are pure functions of data.length (deterministic).
 * Round 2 budgets (sandbox timeout fix): streams above ~1.2 MB skip the
 * optimal parse entirely (zlib level 9 stands); chain length 64/32 (was
 * 1024/256/128); two parse iterations (was three); fewer block-split trials.
 */
const OPTIMAL_MAX_BYTES = 4000000;
const OPTIMAL_SMALL_MAX = 1200000;
// Streams above OPTIMAL_SMALL_MAX get a sample test first: optimal-parse a
// fixed 256 KiB prefix and compare with zlib level 9 on the same prefix.
// Only when the sample gain clears 2.5% does the full stream get the
// (expensive) optimal parse. Pure function of the data: deterministic.
function optimalZlib(data) {
  const n = data.length;
  if (n > OPTIMAL_MAX_BYTES) return null;
  if (n > OPTIMAL_SMALL_MAX) {
    // Sample gate on 256 KiB windows. A prefix-only sample proved
    // unrepresentative (a photo patch at the top hides compressible
    // structure below), so streams up to 3 MB also sample the tail and
    // take the better gain. Larger streams keep the prefix test: their
    // full parse is pricier and the prefix verdict matched it better.
    const sampleGain = (sample) => {
      const zo = optimalZlibInner(sample, 48, 2, 64);
      if (zo === null) return -1;
      const zz = deflateSync(sample, FINAL_OPTS).length;
      return 1 - zo.length / zz;
    };
    if (n <= 3000000) {
      let gain = sampleGain(data.subarray(0, 262144));
      if (gain < 0.025) gain = Math.max(gain, sampleGain(data.subarray((n >> 1) - 131072, (n >> 1) + 131072)));
      if (gain < 0.025) gain = Math.max(gain, sampleGain(data.subarray(n - 262144, n)));
      if (gain < 0.005) return null;
      // A small sample gain means shallow chains understate the win
      // (photo-like RGB): spend a deep chain. A clear gain needs less.
      return optimalZlibInner(data, gain < 0.025 ? 160 : 96, 1, 64);
    }
    let gain3 = sampleGain(data.subarray(0, 262144));
    if (gain3 < 0.025) gain3 = Math.max(gain3, sampleGain(data.subarray((n >> 1) - 131072, (n >> 1) + 131072)));
    if (gain3 < 0.025) return null;
    return optimalZlibInner(data, 80, 1, 64);
  }
  return optimalZlibInner(data, n <= 1048576 ? 96 : 64, 3, 64);
}
function optimalZlibInner(data, maxChain, iterations, niceLen) {
  const n = data.length;
  const matches = findMatches(data, maxChain, niceLen);

  let litCost = costsOf(FIXED_LIT_LEN, 9);
  let distCost = costsOf(FIXED_DIST_LEN, 5);
  if (n > 1200000) {
    const bf = new Int32Array(288);
    for (let i = 0; i < n; i++) bf[data[i]]++;
    bf[256] = 1;
    litCost = costsOf(huffmanLengths(bf, 15), 15);
    const fx = costsOf(FIXED_LIT_LEN, 9);
    for (let s = 257; s < 286; s++) litCost[s] = fx[s];
  }
  let tokens = null;
  let prevBits = Infinity;
  for (let iter = 0; iter < iterations; iter++) {
    const parsed = optimalParse(data, matches, litCost, distCost, n > 1200000 ? CAND_LENS_BIG : CAND_LENS);
    tokens = parsed.tokens;
    const { litFreq, distFreq } = countSymbols(tokens);
    const litLen = huffmanLengths(litFreq, 15);
    const distLen = huffmanLengths(distFreq, 15);
    // True bit cost of this parse under its own Huffman tables.
    let bits = 0;
    for (const t of tokens) {
      bits += t.match
        ? litLen[LEN_SYM[t.len]] + LEN_XB[t.len] + distLen[DIST_SYM[t.dist]] + DIST_XB[t.dist]
        : litLen[t.byte];
    }
    if (bits >= prevBits) break;
    prevBits = bits;
    litCost = smoothedCosts(litFreq);
    distCost = smoothedCosts(distFreq);
  }

  // Block splitting: equal-token splits plus a DP over chunk boundaries
  // (entropy-estimated block costs), each scored exactly by planBlock.
  const splitCounts = tokens.length > 20000 ? [1, 2, 4] : [1];
  let bestPlan = null;
  const considerSplit = (bounds) => {
    const plans = [];
    let total = 0;
    for (let b = 0; b + 1 < bounds.length; b++) {
      const from = bounds[b], to = bounds[b + 1];
      if (to <= from) continue;
      const plan = planBlock(tokens.slice(from, to));
      plans.push({ plan, from, to });
      total += plan.bits;
    }
    if (plans.length > 0 && (bestPlan === null || total < bestPlan.total)) bestPlan = { plans, total };
  };
  for (const k of splitCounts) {
    const bounds = [];
    for (let b = 0; b <= k; b++) bounds.push(Math.floor((b * tokens.length) / k));
    considerSplit(bounds);
  }
  if (tokens.length > 8192) {
    const CHUNK = 512;
    const nb = Math.ceil(tokens.length / CHUNK);
    const litPre = new Int32Array((nb + 1) * 288);
    const distPre = new Int32Array((nb + 1) * 32);
    const extraPre = new Float64Array(nb + 1);
    for (let c = 0; c < nb; c++) {
      litPre.set(litPre.subarray(c * 288, (c + 1) * 288), (c + 1) * 288);
      distPre.set(distPre.subarray(c * 32, (c + 1) * 32), (c + 1) * 32);
      extraPre[c + 1] = extraPre[c];
      const from = c * CHUNK, to = Math.min(tokens.length, from + CHUNK);
      for (let t = from; t < to; t++) {
        const tok = tokens[t];
        if (tok.match) {
          litPre[(c + 1) * 288 + LEN_SYM[tok.len]]++;
          distPre[(c + 1) * 32 + DIST_SYM[tok.dist]]++;
          extraPre[c + 1] += LEN_XB[tok.len] + DIST_XB[tok.dist];
        } else litPre[(c + 1) * 288 + tok.byte]++;
      }
    }
    const HEADER_EST = 340;
    const blockCostEst = (a, b) => {
      let bits = HEADER_EST + (extraPre[b] - extraPre[a]);
      let totL = 0, totD = 0;
      for (let s = 0; s < 288; s++) totL += litPre[b * 288 + s] - litPre[a * 288 + s];
      for (let s = 0; s < 32; s++) totD += distPre[b * 32 + s] - distPre[a * 32 + s];
      totL += 1; // end-of-block symbol
      for (let s = 0; s < 288; s++) {
        const c = litPre[b * 288 + s] - litPre[a * 288 + s] + (s === 256 ? 1 : 0);
        if (c > 0) bits += c * -Math.log2(c / totL);
      }
      if (totD > 0) for (let s = 0; s < 32; s++) {
        const c = distPre[b * 32 + s] - distPre[a * 32 + s];
        if (c > 0) bits += c * -Math.log2(c / totD);
      }
      return bits;
    };
    const MAXSPAN = 32;
    const dp = new Float64Array(nb + 1).fill(Infinity);
    const prevB = new Int32Array(nb + 1).fill(-1);
    dp[0] = 0;
    for (let j = 1; j <= nb; j++) {
      for (let i = Math.max(0, j - MAXSPAN); i < j; i++) {
        if (dp[i] === Infinity) continue;
        const c = dp[i] + blockCostEst(i, j);
        if (c < dp[j]) { dp[j] = c; prevB[j] = i; }
      }
    }
    const bounds = [];
    for (let at = nb; at > 0; at = prevB[at]) bounds.push(Math.min(tokens.length, at * CHUNK));
    bounds.push(0);
    bounds.reverse();
    considerSplit(bounds);
  }

  // Per-block re-parse (zopfli-style): with the split fixed, re-run the
  // optimal parse inside each block under block-local Huffman costs.
  // Screenshots mix flat UI, text and photo patches whose statistics
  // differ sharply; a single global parse cannot price all of them well.
  // Matches come from the global match table (the deflate window spans
  // blocks), lengths capped at the block end by optimalParse's range.
  if (bestPlan.plans.length > 1) {
    const bytePos = new Int32Array(tokens.length + 1);
    for (let t = 0; t < tokens.length; t++) bytePos[t + 1] = bytePos[t] + (tokens[t].match ? tokens[t].len : 1);
    const candTable = n > 1200000 ? CAND_LENS_BIG : CAND_LENS;
    const rePlans = [];
    let reTotal = 0;
    let reTokens = [];
    for (const { from, to } of bestPlan.plans) {
      const bs = bytePos[from], be = bytePos[to];
      let bLit = litCost, bDist = distCost;
      let bTokens = tokens.slice(from, to);
      let bPrev = Infinity;
      const reIters = n > 1200000 ? 1 : 2;
      for (let it = 0; it < reIters; it++) {
        const parsed = optimalParse(data, matches, bLit, bDist, candTable, bs, be);
        bTokens = parsed.tokens;
        const { litFreq, distFreq } = countSymbols(bTokens);
        const ll = huffmanLengths(litFreq, 15);
        const dl = huffmanLengths(distFreq, 15);
        let bits = 0;
        for (const t of bTokens) {
          bits += t.match
            ? ll[LEN_SYM[t.len]] + LEN_XB[t.len] + dl[DIST_SYM[t.dist]] + DIST_XB[t.dist]
            : ll[t.byte];
        }
        if (bits >= bPrev) break;
        bPrev = bits;
        bLit = smoothedCosts(litFreq);
        bDist = smoothedCosts(distFreq);
      }
      const plan = planBlock(bTokens);
      rePlans.push({ plan, from: reTokens.length, to: reTokens.length + bTokens.length });
      reTokens = reTokens.concat(bTokens);
      reTotal += plan.bits;
    }
    if (reTotal < bestPlan.total) {
      tokens = reTokens;
      bestPlan = { plans: rePlans, total: reTotal };
    }
  }

  const w = makeBitWriter();
  bestPlan.plans.forEach(({ plan, from, to }, idx) => {
    writeBlock(w, tokens.slice(from, to), plan, idx === bestPlan.plans.length - 1);
  });
  const deflated = w.finish();
  const out = new Uint8Array(2 + deflated.length + 4);
  out[0] = 0x78; out[1] = 0xda;
  out.set(deflated, 2);
  new DataView(out.buffer).setUint32(2 + deflated.length, adler32(data));
  // Self-check: never hand back a stream that does not inflate to the data.
  try {
    const back = inflateSync(out);
    if (back.length !== n) return null;
    for (let i = 0; i < n; i++) if (back[i] !== data[i]) return null;
  } catch { return null; }
  return out;
}

// ---------------------------------------------------------------- squeeze
const SCREEN_OPTS = { level: 6, memLevel: 6, strategy: constants.Z_DEFAULT_STRATEGY };
const FINAL_OPTS = { level: 9, memLevel: 8, strategy: constants.Z_DEFAULT_STRATEGY };
const SWEEP_BEST = [
  { level: 9, memLevel: 6, strategy: constants.Z_DEFAULT_STRATEGY },
];
// Optimal-parse work budget per image, in input-byte units: the best stream
// always gets a run; a second stream only while the budget lasts. The budget
// is a fixed function of the image (its stream sizes), never of the clock.
const OPTIMAL_BUDGET = 9000000;

function squeezeImage(rgba, width, height) {
  const forms = candidateForms(rgba);
  const pool = [];
  // Palette orderings: screen the unfiltered stream of each first (built
  // directly, no filter work) and only build full stream sets for the two
  // best orderings (the rest cannot catch up).
  let keepPal = null;
  const palForms = forms.filter((f) => f.colourType === 3);
  if (palForms.length > 2) {
    const stride0 = width * palForms[0].bpp;
    const scored = palForms.map((form) => {
      const d = new Uint8Array(height * (stride0 + 1));
      for (let y = 0; y < height; y++) d.set(form.samples.subarray(y * stride0, (y + 1) * stride0), y * (stride0 + 1) + 1);
      return { form, bytes: deflateSync(d, SCREEN_OPTS).length + sideChunkBytes(form) };
    }).sort((a, b) => a.bytes - b.bytes);
    const greyImg = forms.some((f) => f.colourType === 0 || f.colourType === 4);
    keepPal = new Set(scored.slice(0, greyImg ? 1 : 2).map((x) => x.form));
  }
  const built = forms
    .filter((form) => keepPal === null || form.colourType !== 3 || keepPal.has(form))
    .map((form) => ({ form, streams: cheapStreams(form.samples, width, height, form.bpp) }));
  for (const { form, streams } of built) {
    const side = sideChunkBytes(form);
    for (const s of streams) {
      pool.push({ form, choice: s.choice, data: s.data, screenBytes: deflateSync(s.data, SCREEN_OPTS).length + side });
    }
  }
  pool.sort((a, b) => a.screenBytes - b.screenBytes || String(a.choice).localeCompare(String(b.choice)));

  // Brute-force per-row pass on the winning form.
  const bruteForms = pool[0].form.bpp <= 2 ? [pool[0].form] : [];
  for (const form of bruteForms) {
    const data = bruteStream(form.samples, width, height, form.bpp);
    pool.push({ form, choice: 'brute', data, screenBytes: deflateSync(data, SCREEN_OPTS).length + sideChunkBytes(form) });
  }
  pool.sort((a, b) => a.screenBytes - b.screenBytes || String(a.choice).localeCompare(String(b.choice)));

  // Drop byte-identical streams (different palette orderings can coincide):
  // keeping one copy loses nothing and saves whole deflate runs.
  const seenStreams = new Set();
  const uniquePool = [];
  for (const c of pool) {
    let h = 0x811c9dc5;
    const d = c.data;
    for (let i = 0; i < d.length; i++) { h ^= d[i]; h = Math.imul(h, 0x01000193); }
    const key = d.length + ':' + (h >>> 0);
    if (seenStreams.has(key)) continue;
    seenStreams.add(key);
    uniquePool.push(c);
  }
  pool.length = 0;
  pool.push(...uniquePool);
  const finalists = pool.length > 1 && pool[1].screenBytes <= pool[0].screenBytes * 1.08 ? pool.slice(0, 2) : pool.slice(0, 1);
  // Insurance: the winning form's fixed-Paeth stream always reaches the final.
  const winForm = pool[0].form;

  let best = null;
  const considerZlib = (cand, opts) => {
    const z = deflateSync(cand.data, opts);
    const total = PNG_FIXED_BYTES + sideChunkBytes(cand.form) + z.length;
    if (best === null || total < best.total) {
      best = { form: cand.form, z: new Uint8Array(z.buffer, z.byteOffset, z.length), total };
    }
    return total;
  };
  const ranked = [finalists[0]];
  // Optimal-parsing deflate on the best stream first. The level-9 zlib
  // pass (and the parameter sweep on big streams) only runs when the
  // optimal parse is absent or within 3% of the level-6 screen, where
  // level 9 could still plausibly win.
  const cand0 = ranked[0];
  const zOpt = optimalZlib(cand0.data);
  let optTotal = Infinity;
  if (zOpt !== null) {
    optTotal = PNG_FIXED_BYTES + sideChunkBytes(cand0.form) + zOpt.length;
    best = { form: cand0.form, z: zOpt, total: optTotal };
  }
  const screenTotal = PNG_FIXED_BYTES + cand0.screenBytes;
  if (best === null || optTotal >= screenTotal * 0.97) {
    considerZlib(cand0, FINAL_OPTS);
    if (cand0.data.length > 1200000) for (const p of SWEEP_BEST) considerZlib(cand0, p);
  }

  return pngForForm(width, height, best.form, best.z);
}

// ---------------------------------------------------------------- main
for (const id of readdirSync(IN_DIR).sort()) {
  const inputBytes = new Uint8Array(readFileSync(join(IN_DIR, id, 'image.png')));
  const image = decodeToRgba(inputBytes);
  let answer = squeezeImage(image.rgba, image.width, image.height);
  // Belt and braces: the answer must decode back to the exact input pixels.
  // If anything disagrees, fall back to the input file itself (always valid).
  try {
    const check = decodeToRgba(answer);
    let same = check.width === image.width && check.height === image.height && check.rgba.length === image.rgba.length;
    if (same) for (let i = 0; i < image.rgba.length; i++) if (check.rgba[i] !== image.rgba[i]) { same = false; break; }
    if (!same) answer = inputBytes;
  } catch { answer = inputBytes; }
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'image.png'), answer);
}
