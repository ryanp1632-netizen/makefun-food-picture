// food-picture-01 entry (Node 22, standard library only).
//
// The picture is entry.png beside this file. It was made offline by render/generate.py with a free,
// open-weight image model running locally (see README). This program copies those exact bytes to
// /output/entry.png, so both grader runs write the same file.
//
// Outside the grader's container, MAKEFUN_OUTPUT_DIR stands in for /output:
//   mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const MAX_BYTES = 1048576; // the gate's 1 MiB cap

const picture = readFileSync(new URL('./entry.png', import.meta.url));

const isPng = picture.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
const isJpeg = picture[0] === 0xff && picture[1] === 0xd8 && picture.at(-2) === 0xff && picture.at(-1) === 0xd9;
if (!isPng && !isJpeg) throw new Error('entry.png is neither a PNG nor a whole JPEG');
if (picture.length > MAX_BYTES) throw new Error(`entry.png is ${picture.length} bytes, over the ${MAX_BYTES} cap`);

const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
writeFileSync(join(outDir, 'entry.png'), picture);
