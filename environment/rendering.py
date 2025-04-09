# environment/rendering.py
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Colors
COLORS = {
    'agent': (0.0, 0.0, 1.0),
    'waste': (0.55, 0.27, 0.07),
    'bin': (0.2, 0.7, 0.2),
    'ground': (0.9, 0.9, 0.9),
}

def draw_cube(x, y, z, size, color):
    """Draws a colored cube centered at (x, y, z)."""
    glColor3f(*color)
    half = size / 2
    vertices = [
        (x - half, y - half, z - half),
        (x + half, y - half, z - half),
        (x + half, y + half, z - half),
        (x - half, y + half, z - half),
        (x - half, y - half, z + half),
        (x + half, y - half, z + half),
        (x + half, y + half, z + half),
        (x - half, y + half, z + half)
    ]

    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]

    glBegin(GL_QUADS)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_ground(grid_size):
    # Draw base ground
    glColor3f(*COLORS['ground'])
    glBegin(GL_QUADS)
    glVertex3f(0, 0, 0)
    glVertex3f(grid_size, 0, 0)
    glVertex3f(grid_size, 0, grid_size)
    glVertex3f(0, 0, grid_size)
    glEnd()

    # Draw grid lines
    glColor3f(0.2, 0.2, 0.2)
    glLineWidth(1)
    glBegin(GL_LINES)
    for i in range(grid_size + 1):  # Draw lines for all cells including outer edge
        i_f = float(i)
        # Vertical lines (along Z axis)
        glVertex3f(i_f, 0.01, 0.0)
        glVertex3f(i_f, 0.01, float(grid_size))
        
        # Horizontal lines (along X axis)
        glVertex3f(0.0, 0.01, i_f)
        glVertex3f(float(grid_size), 0.01, i_f)
    glEnd()

def setup_camera(grid_size):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(grid_size / 2, grid_size, grid_size * 1.5,  # Eye position
              grid_size / 2, 0, grid_size / 2,            # Look at
              0, 1, 0)                                    # Up vector

def render_waste_env(env, screen):
    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    setup_camera(env.grid_size)
    draw_ground(env.grid_size)

    size = 0.5

    # Draw agent
    ax, ay = env.agent_pos
    draw_cube(ax + 0.5, size / 2, ay + 0.5, size, COLORS['agent'])

    # Draw waste if not carrying
    if not env.carrying_waste:
        wx, wy = env.waste_pos
        draw_cube(wx + 0.5, size / 2, wy + 0.5, size, COLORS['waste'])

    # Draw bin
    bx, by = env.bin_pos
    draw_cube(bx + 0.5, size / 2, by + 0.5, size, COLORS['bin'])

    pygame.display.flip()
    pygame.time.wait(100)
