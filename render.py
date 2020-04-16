#!/usr/bin/env python3

#from math import sin, cos, pi
from random import seed, random, shuffle
seed(0) # <----------- seed ------------

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
st_THIck = [style.linewidth.THIck]
st_THICk = [style.linewidth.THICk]
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
#numpy.random.seed(0)
from bruhat.morse import Assembly, Chain, Flow, Field
from bruhat import element
from bruhat.solve import parse


def conv(x0, y0, x1, y1, a):
    x = (1-a)*x0 + a*x1
    y = (1-a)*y0 + a*y1
    return (x, y)


def arrow(x0, y0, x1, y1, a=0.8, extra=[]):
    x2, y2 = conv(x0, y0, x1, y1, a)
    cvs.stroke(path.line(x0, y0, x2, y2), [deco.earrow()]+extra)
    if a < 1.0:
        cvs.stroke(path.line(x0, y0, x1, y1), extra)


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
    for (dx, dy) in [
        ( 1,  0), (-1,  0), ( 0,  1), ( 0, -1)]:
    #for dx in [-1, 0, 1]:
    #  for dy in [-1, 0, 1]:
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
            arrow(x0, y0, x1, y1, 0.8, [green]+st_thick)

        for pair in matches:
            src, tgt = pair
            x0, y0 = layout[src]
            x1, y1 = layout[tgt]
            arrow(x0, y0, x1, y1, 0.8, [orange]+st_thick)

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


def rescale(chain, scale = 2.0):
    for cell in chain.get_cells():
        if cell.pos is None:
            continue
        x, y = cell.pos
        cell.pos = scale*x, scale*y


def draw_complex(chain, flow, scale=2.0, labels=True):

    rescale(chain, scale)

    r = 0.07

    critical = flow.get_critical()
    A1 = chain.get_bdymap(1) # 1 --> 0
    A2 = chain.get_bdymap(2) # 2 --> 1
    verts = chain.get_cells(0)
    edges = chain.get_cells(1)
    faces = chain.get_cells(2)

    st_arrow = st_THICk+[orange, deco.earrow.Large]

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
    for f in faces:
        bdy = bdys[f]
        assert bdy

        # find all extremal vertices of this Face
        allpos = []
        for e in bdy:
            vs = bdys[e]
            allpos += [v.pos for v in vs]
            if len(vs)==1: # half an edge (only one vertex)
                assert e.pos is not None
                allpos.append(e.pos)
        allpos = list(set(allpos)) # uniq

        # now order these points for drawing
        hull = [allpos[0]]
        while len(hull) < len(allpos):
            vs = [pos for pos in allpos if pos not in hull]
            x0, y0 = hull[-1]
            vs.sort(key = lambda pos : abs(pos[0]-x0)+abs(pos[1]-y0))
            hull.append(vs[0])
        f.hull = hull


        # center
        x0 = sum([pos[0] for pos in hull]) / len(hull)
        y0 = sum([pos[1] for pos in hull]) / len(hull)

        if f.pos == None:
            f.pos = x0, y0

    # draw the infty face first, this is the back face
    # stretched outside to infty
    for f in faces:
        if not f.infty:
            continue
        a = 1.1
        hull = [(a*x,a*y) for (x,y) in f.hull]
        R = 8*r
        x0 = min(x for (x,y) in hull) - R
        x1 = max(x for (x,y) in hull) + R
        y0 = min(y for (x,y) in hull) - R
        y1 = max(y for (x,y) in hull) + R
        cvs.fill(path.rect(x0, y0, x1-x0, y1-y0), [grey])
        dopath(hull, fill=[white], stroke=False)

    # draw the other faces
    matches = flow.get_pairs(1)
    #print("matches:", len(matches))
    for f in faces:
        bdy = bdys[f]

        x0, y0 = f.pos
        if not f.infty:
            hull = [conv(x, y, x0, y0, 0.2) for (x,y) in f.hull]
            dopath(hull, fill=[grey], stroke=False)

        for e in bdy:
            if (e, f) in matches:
                x1, y1 = e.pos
                if f.infty:
                    x0 = 2*x1
                    y0 = 2*y1
                    f.pos = x0, y0
                try:
                    arrow(x1, y1, x0, y0, 1.0, st_arrow)
                except:
                    cvs.text(x0, y0, "???", center)

        if f in critical:
            cvs.stroke(path.circle(f.pos[0], f.pos[1], 2*r), [red]+st_thick)

    # Draw Edges
    matches = flow.get_pairs(0)
    for cell in edges:
        bdy = bdys[cell]
        assert len(bdy) in [1, 2]
        v0 = bdy[0]
        if len(bdy)==2:
            v1 = bdy[1]
            if (v1, cell) in matches:
                v0, v1 = v1, v0 # swap
            x1, y1 = v1.pos
        else:
            x1, y1 = cell.pos
        x0, y0 = v0.pos
        x2, y2 = cell.pos

        if (v0, cell) in matches:
            arrow(x0, y0, x2, y2, 1.0, st_arrow)

        elif cell in critical:
            cvs.stroke(path.circle(x2, y2, 2*r), [red]+st_thick)

        #else:
        #    print("WARNING: not matched & not critical", cell)

        cvs.stroke(path.line(x0, y0, x1, y1))

        if labels:
            name = "$%s$"%cell.name
            annotate(x2, y2, 5*r, name)


    # Draw Vertices
    for cell in verts:
        x, y = cell.pos
    
        cvs.fill(path.circle(x, y, 2*r), [white])
        if cell in critical:
            cvs.stroke(path.circle(x, y, 2*r), [red]+st_thick)
        cvs.fill(path.circle(x, y, r))

        if labels:
            name = "$%s$"%cell.name
            annotate(x, y, 5*r, name)


def draw_matrix(M, flow=None):

    st_text = center

    rows = M.rows
    cols = M.cols

    dx = 0.5
    dy = 0.5
    H = len(rows)*dy
    W = len(cols)*dx

    layout = {}

    for i, col in enumerate(cols):
        x, y = i*dx, H+dy
        layout[col] = x, y
        cvs.text(x, y, "$%s$"%col, st_text)

    for j, row in enumerate(rows):
        x, y = -dx, H-j*dy
        layout[row] = x, y
        cvs.text(x, y, "$%s$"%row, st_text)

    cvs.stroke(path.line(-1.4*dx, H+0.5*dy, W-0.5*dx, H+0.5*dy))
    cvs.stroke(path.line(-0.5*dx, 0.6*dy, -0.5*dx, H+1.4*dy))

    for i, col in enumerate(cols):
      for j, row in enumerate(rows):
        value = M[row, col]
        if value == 0:
            c = '.'
        else:
            c = str(value)
        x, y = i*dx, H-j*dy
        layout[row, col] = x, y
        cvs.text(x, y, "$%s$"%c, st_text)

    if flow is None:
        return

    r = 0.2
    critical = flow.get_critical()
    for item in rows+cols:
        if item in critical:
            x, y = layout[item]
            cvs.stroke(path.circle(x, y, r), st_thick+[red])

    for key in flow.get_pairs():
        if key not in layout:
            continue
        x, y = layout[key]
        cvs.stroke(path.circle(x, y, r), st_thick+[orange])
        



# ---------------------------------------------------------

ring = element.FiniteField(2)


if 0:
    # ------------------------------------ 
    
    M = parse("""
    11.....
    1.1....
    .1.11..
    ..11.1.
    ....1.1
    .....11
    """)
    #print(M)
    M = numpy.array([
        [ 1, 1, 0, 0, 0, 0, 0],
        [ 1, 0, 1, 0, 0, 0, 0],
        [ 0, 1, 0, 1, 1, 0, 0],
        [ 0, 0, 1, 1, 0, 1, 0],
        [ 0, 0, 0, 0, 1, 0, 1],
        [ 0, 0, 0, 0, 0, 1, 1] 
    ])
    
    chain = Chain.fromnumpy(M, ring)
    verts = chain.cells[0]
    for i, v in enumerate(verts):
        x = (i//2)
        y = -(i%2)
        v.pos = x, y
    
    flow = Flow(chain)
    flow.add_match(0, 1, 1)
    flow.add_match(0, 3, 2)
    flow.add_match(0, 4, 4)
    flow.add_match(0, 5, 5)
    flow.add_match(0, 6, 6)
    #flow.build()
    
    #field = Field(chain)
    #field.clamp((0, 1), 1.0)
    #field.clamp((0, 6), 0.0)
    #flow = field.get_flow()
    
    cvs = canvas.canvas()
    draw_matrix(chain.get_bdymap(1), flow)
    save("pic-matrix-graph")
    
    cvs = canvas.canvas()
    draw_complex(chain, flow, labels=True)
    save("pic-complex-graph")
    
    cvs = canvas.canvas()
    draw_poset(chain, flow)
    save("pic-poset-graph")
    
    
    # ------------------------------------ 
    
    chain = Assembly.build_tetrahedron(ring).get_chain()
    
    field = Field(chain)
    field.clamp((0, 1), 0.0)
    field.clamp((2, 4), 1.0)
    flow = field.get_flow()
    
    #flow = Flow(chain)
    #flow.build()

    cvs = canvas.canvas()
    draw_matrix(chain.get_bdymap(2), flow)
    save("pic-matrix-tetra")
    
    cvs = canvas.canvas()
    draw_complex(chain, flow, labels=True)
    save("pic-complex-tetra")
    
    cvs = canvas.canvas()
    draw_poset(chain, flow)
    save("pic-poset-tetra")
    
    
# ------------------------------------ 

#chain = Assembly.build_torus(ring, 2, 2).get_chain() # ugh...

# ------------------------------------ 

m, n = 7, 6
#m, n = 3, 2

ambly = Assembly.build_surface(
    ring, (0, 0), (m, n), 
    open_top=True, open_bot=True)

chain = ambly.get_chain()
for grade in [1, 2]:
    print(chain.get_bdymap(grade))


if 0:
    field = Field(chain)
    for cell in field.cells:
        if cell.grade!=1:
            continue
        i, j, k = cell.key
        if k=="v" and j in [0, n-1]:
            field.clamp(cell, 1.)
        if k=="v" and i in [0, m-2]:
            field.clamp(cell, 0.)

    #field.show() # TODO
    flow = field.get_flow()




flow = Flow(chain)
add_match = flow.add_match

if 0:
    for j in range(n):
        add_match(0, (1,j), (0,j,'v')) # top
        add_match(0, (m-2,j), (m-2,j,'v')) # bot
        if j < n-1:
            add_match(1, (1,j,"h"), (0,j)) # top
            add_match(1, (m-2,j,"h"), (m-2,j)) # bot
    
    for i in range(1, m-2):
        add_match(1, (i, 0, 'v'), (i, 0)) # left
        add_match(1, (i, n-1, 'v'), (i, n-2)) # right
        if 1<i:
            add_match(0, (i,0), (i,0,'h'))
            add_match(0, (i,n-1), (i,n-2,'h'))


cvs = canvas.canvas()
draw_complex(chain, flow, labels=True)
save("pic-complex-surface")





