#!/usr/bin/env python3


import sys, os
from math import *
from random import *

seed(0)

from pyx import canvas, path, deco, trafo, style, text, color, deformer
from pyx.color import rgb
rgbhex = color.rgbfromhexstring

from turtle import Turtle


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
    cvs.stroke(path.line(x0, y0, x1, y1), st_thick+extra)


def draw_poset(chain, flow):

    dx = 1.0
    dy = 2.0
    r = 0.06

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
        cvs.text(xmin-3*r, y, r"$X_%d$"%grade, east)
    
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


def draw_complex(chain, flow):

    r = 0.07

    critical = flow.get_critical()
    A = chain.get_bdymap(1) # 1 --> 0
    verts = chain.get_cells(0)
    edges = chain.get_cells(1)
    matches = flow.get_pairs(0)

    # Edges
    for e in edges:
        bdy = []
        for v in verts:
            if A[v, e]:
                bdy.append(v)
        if len(bdy) != 2:
            continue

        v0, v1 = bdy
        if (v1, e) in matches:
            v0, v1 = v1, v0 # swap
        x0, y0 = v0.pos
        x1, y1 = v1.pos

        x2, y2 = conv(x0, y0, x1, y1, 0.5)
        if (v0, e) in matches:
            arrow(x0, y0, x2, y2, 1.1, st_THick+[orange])

        else:
            assert e in critical
            #cvs.stroke(path.line(x0, y0, x1, y1), st_THick+[red])
            cvs.stroke(path.circle(x2, y2, r), [red]+st_thick)

        cvs.stroke(path.line(x0, y0, x1, y1))

    # Vertices
    for cell in verts:
        x, y = cell.pos
    
        cvs.fill(path.circle(x, y, 2*r), [white])
        if cell in critical:
            cvs.stroke(path.circle(x, y, 2*r), [red]+st_thick)
        cvs.fill(path.circle(x, y, r))



# ---------------------------------------------------------

dx = 2.0
dy = 2.0

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
chain = Chain.fromnumpy(M, ring)
verts = chain.cells[0]
for i, v in enumerate(verts):
    x = (i//2)
    y = -(i%2)
    v.pos = (dx*x, dy*y)


#chain = Assembly.build_tetrahedron(ring).get_chain()
#chain = Assembly.build_torus(2, 2, ring).get_chain()
flow = Flow(chain)
flow.build()


cvs = canvas.canvas()
#draw_poset(chain, flow)
draw_complex(chain, flow)
save("pic-poset-1")




