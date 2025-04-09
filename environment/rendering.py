import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Colors
COLORS = {
    'agent': (0.6, 0.2, 0.8),  
    'waste': (0.55, 0.27, 0.07),  # Brown
    'bin': (0.2, 0.7, 0.2),  # Green
    'ground': (0.9, 0.9, 0.9),  # Light grey
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

def draw_cone(x, y, z, base=0.25, height=0.5, slices=20, color=(0.6, 0.2, 0.8)):
    """Draw a cone (agent head) pointing upward."""
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(-90, 1, 0, 0)  # Rotate to point upward
    quad = gluNewQuadric()
    gluCylinder(quad, base, 0.0, height, slices, 1)
    glPopMatrix()

def draw_sphere(x, y, z, radius=0.25, color=(0.5, 0.5, 0.5)):
    """Draw a solid sphere (waste)."""
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(x, y, z)
    quad = gluNewQuadric()
    gluSphere(quad, radius, 20, 20)
    glPopMatrix()

def draw_cylinder(x, y, z, radius=0.3, height=0.5, slices=20, color=(0.2, 0.7, 0.2)):
    """Draw an open-top cylinder (bin)."""
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(x, y, z)
    glRotatef(-90, 1, 0, 0)
    quad = gluNewQuadric()
    gluCylinder(quad, radius, radius, height, slices, 1)
    glPopMatrix()

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

    # Draw agent: Cube body + cone head
    ax, ay = env.agent_pos
    body_center = (ax + 0.5, size / 2, ay + 0.5)
    draw_cube(*body_center, size, COLORS['agent'])  # Body
    draw_cone(body_center[0], body_center[1] + size / 2, body_center[2], color=(1, 0, 0))  # Head

    # Draw waste
    if not env.carrying_waste:
        wx, wy = env.waste_pos
        draw_sphere(wx + 0.5, size / 2, wy + 0.5, radius=0.25, color=COLORS['waste'])

    # Draw bin
    bx, by = env.bin_pos
    draw_cylinder(bx + 0.5, size / 2, by + 0.5, radius=0.3, height=0.5, color=COLORS['bin'])

    pygame.display.flip()
    pygame.time.wait(100)
