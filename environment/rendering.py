# environment/rendering.py
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# Color definitions
COLORS = {
    'background': (0.95, 0.95, 0.95),
    'grid': (0.8, 0.7, 0.9),
    'agent': (0.0, 0.0, 1.0),      # Blue
    'waste': (0.55, 0.27, 0.07),   # Brownish waste
    'bin': (0.0, 1.0, 0.0),        # Green bin
    'house_wall': (0.8, 0.5, 0.2), # Wall color for houses
    'house_roof': (0.6, 0.1, 0.1)  # Roof color for houses
}

def init_gl(screen):
    """Initialize OpenGL for 3D rendering."""
    glViewport(0, 0, screen.get_width(), screen.get_height())
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    aspect = screen.get_width() / screen.get_height()
    gluPerspective(45, aspect, 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def setup_camera(env):
    """Position the camera for an elevated view of the ground-level scene."""
    grid_size = env.grid_size
    cam_x = grid_size * 1.2
    cam_y = grid_size * 1.5  # Elevated view
    cam_z = grid_size * 1.2
    center = (grid_size / 2.0, 0, grid_size / 2.0)
    up = (0, 1, 0)
    gluLookAt(cam_x, cam_y, cam_z,
              center[0], center[1], center[2],
              up[0], up[1], up[2])

def draw_cube(center, size, color):
    """Draws a solid cube centered at 'center'."""
    glColor3f(*color)
    hs = size / 2.0
    vertices = [
        (center[0]-hs, center[1]-hs, center[2]-hs),
        (center[0]+hs, center[1]-hs, center[2]-hs),
        (center[0]+hs, center[1]+hs, center[2]-hs),
        (center[0]-hs, center[1]+hs, center[2]-hs),
        (center[0]-hs, center[1]-hs, center[2]+hs),
        (center[0]+hs, center[1]-hs, center[2]+hs),
        (center[0]+hs, center[1]+hs, center[2]+hs),
        (center[0]-hs, center[1]+hs, center[2]+hs)
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4)
    ]
    glBegin(GL_QUADS)
    for face in faces:
        for vertex in face:
            glVertex3f(*vertices[vertex])
    glEnd()

def draw_sphere(center, radius, color, slices=16, stacks=16):
    """Draw a solid sphere with the given center and radius."""
    glPushMatrix()
    glTranslatef(center[0], center[1], center[2])
    glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)
    glPopMatrix()

def draw_grid(env):
    """Draw a grid floor on the x-z plane at y = 0."""
    grid_size = env.grid_size
    glColor3f(*COLORS['grid'])
    glLineWidth(1)
    glBegin(GL_LINES)
    for i in range(grid_size + 1):
        glVertex3f(i, 0, 0)
        glVertex3f(i, 0, grid_size)
        glVertex3f(0, 0, i)
        glVertex3f(grid_size, 0, i)
    glEnd()

def draw_house(center, base_size, wall_color, roof_color):
    """
    Draw a simple house with cube walls and a pyramid roof.
    The 'center' here refers to the center of the house's base.
    """
    # Draw walls: adjust cube so its bottom is at y = 0.
    wall_center = (center[0], base_size/2, center[2])
    draw_cube(wall_center, base_size, wall_color)
    
    # Pyramid roof
    hs = base_size / 2.0
    roof_height = base_size / 2.0
    v0 = (center[0]-hs, base_size, center[2]-hs)
    v1 = (center[0]+hs, base_size, center[2]-hs)
    v2 = (center[0]+hs, base_size, center[2]+hs)
    v3 = (center[0]-hs, base_size, center[2]+hs)
    apex = (center[0], base_size + roof_height, center[2])
    
    glColor3f(*roof_color)
    glBegin(GL_TRIANGLES)
    glVertex3f(*apex)
    glVertex3f(*v0)
    glVertex3f(*v1)
    
    glVertex3f(*apex)
    glVertex3f(*v1)
    glVertex3f(*v2)
    
    glVertex3f(*apex)
    glVertex3f(*v2)
    glVertex3f(*v3)
    
    glVertex3f(*apex)
    glVertex3f(*v3)
    glVertex3f(*v0)
    glEnd()

def render_waste_env(env, screen=None):
    """
    Renders the 3D environment. This function now draws buildings/houses using
    the dynamically generated positions from the environment.
    """
    if screen is None:
        screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    init_gl(screen)
    glClearColor(*COLORS['background'], 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glLoadIdentity()
    setup_camera(env)

    # Draw the grid floor.
    draw_grid(env)

    # --- Draw Dynamic Houses (Buildings) ---
    houses = getattr(env, 'houses', None)
    if houses is None:
        # Fallback: static houses (if env.houses is not set).
        houses = [
            {'pos': np.array([1, 0, 1], dtype=np.int32), 'size': 1.5},
            {'pos': np.array([4, 0, 2], dtype=np.int32), 'size': 2.0},
            {'pos': np.array([2, 0, 6], dtype=np.int32), 'size': 1.8}
        ]
    for house in houses:
        house_center = (house['pos'][0] + 0.5, 0, house['pos'][2] + 0.5)
        draw_house(house_center, house['size'], COLORS['house_wall'], COLORS['house_roof'])
    
    # Helper to get centers for cubes and spheres so that objects sit on the ground.
    def get_cube_center(pos, cube_size):
        return (pos[0] + 0.5, cube_size / 2, pos[2] + 0.5)
    
    def get_sphere_center(pos, radius):
        return (pos[0] + 0.5, radius, pos[2] + 0.5)
    
    # --- Draw Interactive Objects ---
    # Bin (cube of size 0.8)
    bin_center = get_cube_center(env.bin_pos, 0.8)
    draw_cube(bin_center, 0.8, COLORS['bin'])
    
    # Waste (sphere of radius 0.35), if not being carried.
    if not env.carrying_waste:
        waste_center = get_sphere_center(env.waste_pos, 0.35)
        draw_sphere(waste_center, 0.35, COLORS['waste'])
    
    # Agent (cube of size 0.7)
    agent_center = get_cube_center(env.agent_pos, 0.7)
    draw_cube(agent_center, 0.7, COLORS['agent'])
    
    pygame.display.flip()
    pygame.time.wait(100)
