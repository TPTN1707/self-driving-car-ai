import math
import pygame
from src.config import (
    COLOR_WHITE,
    CAR_HEIGHT,
    CAR_SPEED,
    CAR_WIDTH,
    COLOR_BLACK,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    GAME_SPEED,
    ROTATION_SPEED,
    SENSOR_LENGTH,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from src.track import Track

# Initialize Pygame and fonts
pygame.init()
font = pygame.font.SysFont("arial", 25)


def get_intersection(p1, p2, p3, p4):
    """Calculates the intersection point of segment p1-p2 and p3-p4.

    Returns (x, y) if intersection exists, otherwise None.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        # Intersection point coordinate
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        return (px, py)

    return None


class Car:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0  # Heading angle in degrees (0 is facing right)
        self.speed = CAR_SPEED
        self.is_dead = False
        self.corners = []

    def update(self, action):
        """Updates car rotation and position based on action."""
        # Action: [straight, turn_right, turn_left]
        if action[1] == 1:
            self.angle += ROTATION_SPEED  # Rotate clockwise
        elif action[2] == 1:
            self.angle -= ROTATION_SPEED  # Rotate counter-clockwise

        # Keep angle within 0-359 degrees
        self.angle %= 360

        # Trigonometry: Move forward based on current heading angle
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)

        # Recalculate 4 corner coordinates of the car body
        self._calculate_corners()

    def _calculate_corners(self):
        """Calculates the coordinates of the 4 corners of the rotated car."""
        rad = math.radians(self.angle)
        cos_rad = math.cos(rad)
        sin_rad = math.sin(rad)

        # Half dimensions
        w = CAR_WIDTH / 2
        h = CAR_HEIGHT / 2

        # Vectors relative to center
        front_dx = h * cos_rad
        front_dy = h * sin_rad
        side_dx = w * sin_rad
        side_dy = -w * cos_rad

        # Calculate 4 corners: front-right, front-left, back-left, back-right
        self.corners = [
            (self.x + front_dx + side_dx, self.y + front_dy + side_dy),
            (self.x + front_dx - side_dx, self.y + front_dy - side_dy),
            (self.x - front_dx - side_dx, self.y - front_dy - side_dy),
            (self.x - front_dx + side_dx, self.y - front_dy + side_dy),
        ]

    def get_sensor_readings(self, walls):
        """Casts 5 laser rays and measures distance to closest walls."""
        readings = []
        # 5 sensor angles relative to car heading: [-90, -45, 0, 45, 90] degrees
        sensor_angles = [-90, -45, 0, 45, 90]

        for rel_angle in sensor_angles:
            abs_angle = math.radians(self.angle + rel_angle)
            # End point of the laser ray if it doesn't hit anything
            dest_x = self.x + SENSOR_LENGTH * math.cos(abs_angle)
            dest_y = self.y + SENSOR_LENGTH * math.sin(abs_angle)
            ray_end = (dest_x, dest_y)

            closest_point = None
            min_dist = SENSOR_LENGTH

            # Check intersection of this ray with EVERY wall of the track
            for wall in walls:
                intersect = get_intersection((self.x, self.y), ray_end, wall[0], wall[1])
                if intersect:
                    # Calculate distance to intersection point
                    dist = math.hypot(intersect[0] - self.x, intersect[1] - self.y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = intersect

            readings.append((min_dist, closest_point, ray_end))

        return readings

    def draw(self, display):
        """Draws the car body as a polygon connecting its 4 corners."""
        pygame.draw.polygon(display, COLOR_YELLOW, self.corners)


class CarGameAI:

    def __init__(self):
        self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Self-Driving Car AI (DQN)")
        self.clock = pygame.time.Clock()
        self.track = Track()
        self.reset()

    def reset(self):
        """Resets the environment for a new episode."""
        # Spawn the car at the starting point of the track
        self.car = Car(100, 300)
        self.car.angle = 90  # Face downwards to align with the track lane
        self.score = 0
        self.frame_iteration = 0

    def play_step(self, action):
        """Executes one simulation frame based on the agent's action."""
        self.frame_iteration += 1

        # Handle window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 1. Update car physics (movement & rotation)
        self.car.update(action)

        # 2. Extract sensor readings
        sensor_data = self.car.get_sensor_readings(self.track.walls)

        # 3. Check for collisions
        reward = 0.1  # Reward +0.1 for surviving each frame
        game_over = False

        if self._is_collision() or self.frame_iteration > 2000:
            game_over = True
            reward = -10  # Heavy penalty for crashing
            return reward, game_over, self.score

        # Update score based on survival time (frame iteration)
        self.score = self.frame_iteration // 10

        # 4. Render display
        self._update_ui(sensor_data)
        self.clock.tick(GAME_SPEED)

        return reward, game_over, self.score

    def _is_collision(self):
        """Checks if any of the car's 4 edges intersects with any track wall."""
        car_edges = [
            (self.car.corners[0], self.car.corners[1]),
            (self.car.corners[1], self.car.corners[2]),
            (self.car.corners[2], self.car.corners[3]),
            (self.car.corners[3], self.car.corners[0]),
        ]

        for edge in car_edges:
            for wall in self.track.walls:
                if get_intersection(edge[0], edge[1], wall[0], wall[1]):
                    return True
        return False

    def _update_ui(self, sensor_data):
        """Renders the track, car, lasers, and scoreboard."""
        self.display.fill(COLOR_BLACK)

        # Draw the racetrack walls
        self.track.draw(self.display)

        # Draw the active sensor lasers
        for dist, intersect, ray_end in sensor_data:
            if intersect:
                # Draw green line from car to collision point
                pygame.draw.line(self.display, COLOR_GREEN, (self.car.x, self.car.y), intersect, 1)
                # Draw red circle at collision point
                pygame.draw.circle(self.display, COLOR_RED, (int(intersect[0]), int(intersect[1])), 4)
            else:
                # Draw faint green line to max length if no wall hit
                pygame.draw.line(self.display, COLOR_GREEN, (self.car.x, self.car.y), ray_end, 1)

        # Draw the car body
        self.car.draw(self.display)

        # Draw the scoreboard
        text = font.render(f"Score: {self.score}", True, COLOR_WHITE)
        self.display.blit(text, [10, 10])
        pygame.display.flip()