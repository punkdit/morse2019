#!/usr/bin/env python3


import sys, os
from math import *
from random import *

seed(0)

from pyx import canvas, path, deco, trafo, style, text, color, deformer
from pyx.color import rgb
from pyx.bbox import bbox
rgbhex = color.rgbfromhexstring

from turtle import Turtle, dopath


#text.set(mode="latex") 
text.set(cls = text.LatexRunner)
#text.set(docopt="12pt")
text.preamble(r"\usepackage{amsmath,amsfonts,amssymb}")
#text.preamble(r"\LARGE") #??


north = [text.halign.boxcenter, text.valign.top]
northeast = [text.halign.boxright, text.valign.top]
northwest = [text.halign.boxleft, text.valign.top]
south = [text.halign.boxcenter, text.valign.bottom]
southeast = [text.halign.boxright, text.valign.bottom]
southwest = [text.halign.boxleft, text.valign.bottom]
east = [text.halign.boxright, text.valign.middle]
west = [text.halign.boxleft, text.valign.middle]
center = [text.halign.boxcenter, text.valign.middle]

black = rgb(0., 0., 0.) 
blue = rgb(0.1, 0.1, 0.9)
red = lred = rgb(1.0, 0.1, 0.1)
orange = rgbhex("F36B08")
green = rgb(0.0, 0.6, 0.0)
white = rgb(1., 1., 1.) 
#shade = rgb(0.75, 0.55, 0)
grey = rgb(0.85, 0.85, 0.85)
darkgrey = rgb(0.65, 0.65, 0.65)
yellow = rgb(1., 1., 0.)

shade0 = rgbhex("c5e16f") 
shade1 = rgbhex("a8d18a") 
shade2 = rgbhex("539699") 
shade3 = rgbhex("788696") 
#shade4 = rgbhex("f1646c")  # hot pink. :P
shade4 = rgbhex("ea4850")
shade5 = rgbhex("fac174") 

st_dashed = [style.linestyle.dashed]
st_dotted = [style.linestyle.dotted]
st_round = [style.linecap.round]

st_thick = [style.linewidth.thick]
st_Thick = [style.linewidth.Thick]
st_THick = [style.linewidth.THick]
st_THICK = [style.linewidth.THICK]
#print dir(style.linewidth)

st_font = [text.size.Large]



def save(name):
    name = "images/"+name
    cvs.writePDFfile(name)
    cvs.writeSVGfile(name)
    print(name)


stack = []
def push(*args):
    global cvs
    stack.append(cvs)
    cvs = canvas.canvas(*args)

def pop(*args):
    global cvs
    c1 = stack.pop()
    c1.insert(cvs, *args)
    cvs = c1




# ----------------------------------------------------------------

import numpy
from bruhat.morse import Assembly, Chain, Flow
from bruhat import element
from bruhat.solve import parse


def conv(x0, y0, x1, y1, a):
    x = (1-a)*x0 + a*x1
    y = (1-a)*y0 + a*y1
    return (x, y)


def arrow(x0, y0, x1, y1, a=0.8, extra=[]):
    x2, y2 = conv(x0, y0, x1, y1, a)
    cvs.stroke(path.line(x0, y0, x2, y2), [deco.earrow()]+st_thick+extra)
    if a < 1.0:
        cvs.stroke(path.line(x0, y0, x1, y1), st_thick+extra)


def hitlist(x0, y0, x1, y1):
    bb0 = bbox(x0, y0, x1, y1)
    hits = []
    for item in cvs:
        bb1 = item.bbox()
        if bb0.intersects(bb1):
            hits.append(item)
    return hits


def annotate(x, y, r, label): 
    count = None
    best = None
    epsilon = 0.1*r
    #debug = print if label=="$v_{2}$" else lambda *args,**kw:None
    #debug(label)
    #for (dx, dy) in [
    #    ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1)]:
    for dx in [-1, 0, 1]:
      for dy in [-1, 0, 1]:
        x1 = x+r*dx
        y1 = y+r*dy
        #hits = hitlist(x1-r, y1-r, x1+r, y1+r)
        hits = hitlist(x1-epsilon, y1-epsilon, x1+epsilon, y1+epsilon)
        #debug("\t", dx, dy)
        #for hit in hits:
            #debug("\t", hit)
        if count is None or len(hits) < count:
            count = len(hits)
            best = x1, y1
    x1, y1 = best
    #debug("count:", count, label)
    cvs.text(x1, y1, label, center)


def draw_poset(chain, flow, dx=1.0, dy=2.0, r=0.06):

    y = 0.
    layout = {} # map Cell -> (x,y)
    critical = flow.get_critical()
    grades = flow.all_grades()
    xmin = 0.
    xmax = 0.
    for grade in grades:
        cells = chain.cells[grade]
        x = -0.5*dx*len(cells)
        xmin = min(x, xmin)
        for cell in cells:
            layout[cell] = x, y
            x += dx
        xmax = max(x-dx, xmax)
        y += dy

    critical = set(critical)
    for grade in grades:
        y = grade * dy
        cvs.stroke(path.line(xmin, y, xmax, y), [darkgrey])
        cvs.text(xmin-5*r, y, r"$X_%d$"%grade, east)
    
    #grade = 1
    for grade in grades:
        A = chain.get_bdymap(grade) # grade --> grade-1
        #print(A)
        if A.is_zero():
            continue
    
        matches = flow.get_pairs(grade-1) # grade-1 --> grade
        
        for (key, value) in A.elements.items():
            row, col = key
            if (row, col) in matches:
                continue
            x0, y0 = layout[col]
            x1, y1 = layout[row]
            arrow(x0, y0, x1, y1, 0.8, [green])

        for pair in matches:
            src, tgt = pair
            x0, y0 = layout[src]
            x1, y1 = layout[tgt]
            arrow(x0, y0, x1, y1, 0.8, [orange])

    # draw cells last
    for (cell, (x, y)) in layout.items():
        cvs.fill(path.circle(x, y, 2*r), [white])
        if cell in critical:
            cvs.stroke(path.circle(x, y, 2*r), [red]+st_thick)
        cvs.fill(path.circle(x, y, r))

        name = "$%s$"%cell.name
        if cell.grade == 0:
            cvs.text(x, y-3*r, name, north)

        elif cell.grade == grades[-1]:
            cvs.text(x, y+3*r, name, south)

        else:
            cvs.text(x+3*r, y-r, name, northwest)


def draw_complex(chain, flow, scale):

    r = 0.07

    critical = flow.get_critical()
    A1 = chain.get_bdymap(1) # 1 --> 0
    A2 = chain.get_bdymap(2) # 2 --> 1
    verts = chain.get_cells(0)
    edges = chain.get_cells(1)
    faces = chain.get_cells(2)

    bdys = {} # map cell --> list of bdy cells

    for f in faces:
        bdy = []
        for e in edges:
            if A2[e, f]:
                bdy.append(e)
        bdys[f] = bdy

    for e in edges:
        bdy = []
        for v in verts:
            if A1[v, e]:
                bdy.append(v)
        bdys[e] = bdy

    for e in edges:
        bdy = bdys[e]
        if len(bdy) != 2:
            continue
        v0, v1 = bdy
        x0, y0 = v0.pos
        x1, y1 = v1.pos
        x2, y2 = conv(x0, y0, x1, y1, 0.5)
        e.pos = x2, y2

    # Draw Faces
    matches = flow.get_pairs(1)
    for f in faces:
        bdy = bdys[f]
        assert bdy

        hull = []
        for e in bdy:
            hull += bdys[e]
        hull = list(set(hull)) # uniq

        x0 = sum([v.pos[0] for v in hull]) / len(hull)
        y0 = sum([v.pos[1] for v in hull]) / len(hull)

        if f.pos == None:
            f.pos = x0, y0

        if not f.infty:
            hull = [v.pos for v in hull]
            hull = [conv(x, y, x0, y0, 0.2) for (x,y) in hull]
            #p = path.path(path.moveto(*hull[0]),
            #cvs.fill(p, [grey])
            dopath(hull, fill=[grey], stroke=False)

        for e in bdy:
            if (e, f) in matches:
                x1, y1 = e.pos
                if f.infty:
                    x0 = 2*x1
                    y0 = 2*y1
                    f.pos = x0, y0
                arrow(x1, y1, x0, y0, 1.0, st_THick+[orange])

        if f in critical:
            cvs.stroke(path.circle(f.pos[0], f.pos[1], r), [red]+st_thick)

    # Draw Edges
    matches = flow.get_pairs(0)
    for cell in edges:
        bdy = bdys[cell]
        v0, v1 = bdy
        if (v1, cell) in matches:
            v0, v1 = v1, v0 # swap
        x0, y0 = v0.pos
        x1, y1 = v1.pos
        x2, y2 = cell.pos

        if (v0, cell) in matches:
            arrow(x0, y0, x2, y2, 1.0, st_THick+[orange])

        elif cell in critical:
            cvs.stroke(path.circle(x2, y2, r), [red]+st_thick)

        cvs.stroke(path.line(x0, y0, x1, y1))

        name = "$%s$"%cell.name
        annotate(x2, y2, 5*r, name)


    # Draw Vertices
    for cell in verts:
        x, y = cell.pos
    
        cvs.fill(path.circle(x, y, 2*r), [white])
        if cell in critical:
            cvs.stroke(path.circle(x, y, 2*r), [red]+st_thick)
        cvs.fill(path.circle(x, y, r))

        name = "$%s$"%cell.name
        annotate(x, y, 5*r, name)


# ---------------------------------------------------------

M = parse("""
11.....
1.1....
.1.11..
..11.1.
....1.1
.....11
""")

ring = element.FiniteField(2)

#rows = "1 2 3 4 5 6".split()
#cols = "1 2 3 4 5 6 7".split()
chains = []

chain = Chain.fromnumpy(M, ring)
verts = chain.cells[0]
for i, v in enumerate(verts):
    x = (i//2)
    y = -(i%2)
    v.pos = x, y
chains.append(chain)

chain = Assembly.build_tetrahedron(ring).get_chain()
chains.append(chain)

chain = Assembly.build_torus(2, 2, ring).get_chain()
chains.append(chain)

scale = 2.0
for chain in chains:
    for cell in chain.get_cells():
        if cell.pos is None:
            continue
        x, y = cell.pos
        cell.pos = scale*x, scale*y

for idx, chain in enumerate(chains):

#    if idx != 0:
#        continue

    flow = Flow(chain)
    flow.build()
    
    cvs = canvas.canvas()
    draw_complex(chain, flow, scale)

    save("pic-complex-%d"%idx)

    #break
    
    cvs = canvas.canvas()
    draw_poset(chain, flow)
    save("pic-poset-%d"%idx)




