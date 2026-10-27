"""One compiled cell, read whole and written back (knoxlots' header.rs and pack.rs, in Python).

    cell = Cell.read(lots_dir, cx, cy)
    cell.squares[(x, y, z)] = [tile names]     # cell-local x, y (0..255)
    cell.rooms = [(name, z, [(x, y, w, h)], [(a, b, c)])]   # rects cell-local (may run past 255)
    cell.buildings = [[room indexes]]
    cell.write(lots_dir)
"""
from __future__ import annotations

import os
import struct

CELL, CHUNKS, CHUNK = 256, 32, 8


class Cell:
    def __init__(self, cx, cy):
        self.cx, self.cy = cx, cy
        self.version = 1
        self.fields = (8, 8)
        self.min_z = self.max_z = 0
        self.squares: dict = {}
        self.rooms: list = []
        self.buildings: list = []
        self.chunk_values = b"\0" * 1024
        self.pack_version = 1

    # -------------------------------------------------------------- read
    @classmethod
    def read(cls, lots: str, cx: int, cy: int) -> "Cell":
        c = cls(cx, cy)
        d = open(os.path.join(lots, f"{cx}_{cy}.lotheader"), "rb").read()
        assert d[:4] == b"LOTH"
        o = 4
        c.version, n = struct.unpack_from("<2I", d, o); o += 8
        names = []
        for _ in range(n):
            e = d.index(b"\n", o); names.append(d[o:e].decode("latin-1")); o = e + 1
        fa, fb, c.min_z, c.max_z = struct.unpack_from("<4i", d, o); o += 16
        c.fields = (fa, fb)
        (nr,) = struct.unpack_from("<I", d, o); o += 4
        for _ in range(nr):
            e = d.index(b"\n", o); name = d[o:e].decode("latin-1"); o = e + 1
            z, k = struct.unpack_from("<2I", d, o); o += 8
            rects = [struct.unpack_from("<4I", d, o + 16 * j) for j in range(k)]; o += 16 * k
            (m,) = struct.unpack_from("<I", d, o); o += 4
            objs = [struct.unpack_from("<3I", d, o + 12 * j) for j in range(m)]; o += 12 * m
            c.rooms.append([name, z, [tuple(r) for r in rects], [tuple(b) for b in objs]])
        (nb,) = struct.unpack_from("<I", d, o); o += 4
        for _ in range(nb):
            (k,) = struct.unpack_from("<I", d, o); o += 4
            c.buildings.append(list(struct.unpack_from(f"<{k}I", d, o))); o += 4 * k
        c.chunk_values = d[o:o + 1024]
        assert len(d) - o == 1024, (len(d) - o)
        # pack
        p = open(os.path.join(lots, f"world_{cx}_{cy}.lotpack"), "rb").read()
        assert p[:4] == b"LOTP"
        c.pack_version, nch = struct.unpack_from("<2I", p, 4)
        offs = struct.unpack_from(f"<{nch}q", p, 12)
        ends = sorted(set(list(offs) + [len(p)]))
        for ci in range(nch):
            q = offs[ci]
            if q < 0:
                continue
            end = ends[ends.index(q) + 1]
            bx, by = (ci // CHUNKS) * CHUNK, (ci % CHUNKS) * CHUNK
            i = 0
            levels = c.max_z - c.min_z + 1
            while q < end and i < 64 * levels:
                (v,) = struct.unpack_from("<I", p, q); q += 4
                if v == 0xFFFFFFFF:
                    (cnt,) = struct.unpack_from("<I", p, q); q += 4
                    i += cnt
                    continue
                vals = struct.unpack_from(f"<{v}I", p, q); q += 4 * v
                zi, rem = divmod(i, 64)
                x, y = divmod(rem, CHUNK)
                tiles = [names[t] for t in vals[1:]]
                if tiles:
                    c.squares[(bx + x, by + y, c.min_z + zi)] = tiles
                i += 1
        return c

    # -------------------------------------------------------------- write
    def write(self, lots: str) -> None:
        names, index = [], {}
        for t in (t for ts in self.squares.values() for t in ts):
            if t not in index:
                index[t] = len(names); names.append(t)
        zs = [k[2] for k in self.squares] + [r[1] for r in self.rooms]
        self.min_z = min([0] + zs)
        self.max_z = max([0] + zs)
        h = bytearray(b"LOTH") + struct.pack("<2I", self.version, len(names))
        for t in names:
            h += t.encode("latin-1") + b"\n"
        h += struct.pack("<4i", self.fields[0], self.fields[1], self.min_z, self.max_z)
        h += struct.pack("<I", len(self.rooms))
        for name, z, rects, objs in self.rooms:
            h += name.encode("latin-1") + b"\n" + struct.pack("<2I", z, len(rects))
            for r in rects:
                h += struct.pack("<4I", *r)
            h += struct.pack("<I", len(objs))
            for ob in objs:
                h += struct.pack("<3I", *ob)
        h += struct.pack("<I", len(self.buildings))
        for b in self.buildings:
            h += struct.pack("<I", len(b)) + struct.pack(f"<{len(b)}I", *b)
        h += self.chunk_values
        open(os.path.join(lots, f"{self.cx}_{self.cy}.lotheader"), "wb").write(bytes(h))
        # pack
        body = bytearray()
        offs = []
        head_len = 12 + 8 * CHUNKS * CHUNKS
        for ci in range(CHUNKS * CHUNKS):
            bx, by = (ci // CHUNKS) * CHUNK, (ci % CHUNKS) * CHUNK
            recs = bytearray()
            skip = 0
            anything = False
            for z in range(self.min_z, self.max_z + 1):
                for x in range(CHUNK):
                    for y in range(CHUNK):
                        ts = self.squares.get((bx + x, by + y, z))
                        if not ts:
                            skip += 1
                            continue
                        if skip:
                            recs += struct.pack("<2I", 0xFFFFFFFF, skip); skip = 0
                        recs += struct.pack("<I", len(ts) + 1) + struct.pack("<i", -1)
                        recs += struct.pack(f"<{len(ts)}I", *(index[t] for t in ts))
                        anything = True
            if skip:
                recs += struct.pack("<2I", 0xFFFFFFFF, skip)
            offs.append(head_len + len(body))
            body += recs
        p = bytearray(b"LOTP") + struct.pack("<2I", self.pack_version, CHUNKS * CHUNKS)
        p += struct.pack(f"<{len(offs)}q", *offs) + body
        open(os.path.join(lots, f"world_{self.cx}_{self.cy}.lotpack"), "wb").write(bytes(p))
