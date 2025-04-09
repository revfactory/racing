import pygame
import math
import random
import numpy as np
from pygame.locals import *

# --- 기본 설정 ---
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
FPS = 60
LAPS_TO_WIN = 3

# --- 색상 ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
TRACK_COLOR = (100, 100, 100)
GRASS_COLOR = (93, 164, 93)

# --- Pygame 초기화 ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Advanced 3D Racing")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)
small_font = pygame.font.Font(None, 24)

# --- 향상된 3D 시스템 ---
# 행렬 변환 유틸리티 함수
def perspective_matrix(fov, aspect, near, far):
    """투영 행렬 생성"""
    f = 1.0 / math.tan(fov / 2)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0]
    ])

def look_at_matrix(eye, target, up):
    """뷰 행렬 생성"""
    z = normalize(eye - target)  # 카메라가 바라보는 방향의 반대 방향
    x = normalize(np.cross(up, z))
    y = np.cross(z, x)
    
    return np.array([
        [x[0], x[1], x[2], -np.dot(x, eye)],
        [y[0], y[1], y[2], -np.dot(y, eye)],
        [z[0], z[1], z[2], -np.dot(z, eye)],
        [0, 0, 0, 1]
    ])

def normalize(v):
    """벡터 정규화"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def transform_point(matrix, point):
    """점을 변환 행렬에 적용"""
    # 4D 동차 좌표로 변환
    p = np.append(point, 1.0)
    # 행렬 변환 적용
    transformed = np.dot(matrix, p)
    # w로 나누어 다시 3D 좌표로 변환 (원근 나눗셈)
    if transformed[3] != 0:
        transformed = transformed / transformed[3]
    return transformed[:3]

# 카메라 설정
camera = {
    'position': np.array([0.0, 5.0, -20.0]),  # 카메라 위치
    'target': np.array([0.0, 0.0, 0.0]),      # 카메라가 바라보는 지점
    'up': np.array([0.0, 1.0, 0.0]),          # 카메라 상단 방향
    'fov': math.radians(70),                  # 시야각 (라디안)
    'aspect': SCREEN_WIDTH / SCREEN_HEIGHT,    # 화면 비율
    'near': 0.1,                              # 가까운 클리핑 평면
    'far': 1000.0                             # 먼 클리핑 평면
}

def world_to_screen(point_3d):
    """3D 월드 좌표를 2D 화면 좌표로 변환 (향상된 버전)"""
    # 뷰 행렬 계산
    view_matrix = look_at_matrix(camera['position'], camera['target'], camera['up'])
    # 투영 행렬 계산
    proj_matrix = perspective_matrix(camera['fov'], camera['aspect'], camera['near'], camera['far'])
    
    # 뷰 변환
    view_point = transform_point(view_matrix, point_3d)
    
    # 카메라 뒤의 점은 보이지 않음
    if view_point[2] > 0:
        return None
    
    # 투영 변환
    proj_point = transform_point(proj_matrix, view_point)
    
    # NDC 좌표계에서 화면 좌표계로 변환
    screen_x = (proj_point[0] + 1.0) * 0.5 * SCREEN_WIDTH
    screen_y = (1.0 - (proj_point[1] + 1.0) * 0.5) * SCREEN_HEIGHT
    
    # Z 값도 반환 (깊이 정보)
    return (int(screen_x), int(screen_y), view_point[2])

# Z-buffer 구현 (단순화된 버전)
class ZBuffer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.clear()
    
    def clear(self):
        # 최대 깊이로 초기화 (음수 값이 더 가까움)
        self.buffer = np.full((self.height, self.width), float('inf'))
    
    def test_and_set(self, x, y, z):
        """해당 픽셀에 그릴 수 있는지 테스트하고 z 값 업데이트"""
        if 0 <= x < self.width and 0 <= y < self.height:
            if z < self.buffer[y, x]:
                self.buffer[y, x] = z
                return True
        return False

z_buffer = ZBuffer(SCREEN_WIDTH, SCREEN_HEIGHT)

# --- 3D 모델 로딩 및 렌더링 ---
class Mesh:
    """3D 메시를 표현하는 클래스"""
    def __init__(self):
        self.vertices = []  # 정점 목록
        self.faces = []     # 면 목록 (정점 인덱스 참조)
        self.color = WHITE  # 기본 색상
    
    def create_cube(self, size):
        """간단한 큐브 메시 생성"""
        s = size / 2
        # 8개의 정점 (앞면 4개, 뒷면 4개)
        self.vertices = [
            np.array([-s, -s, -s]),  # 0: 좌하단 앞
            np.array([s, -s, -s]),   # 1: 우하단 앞
            np.array([s, s, -s]),    # 2: 우상단 앞
            np.array([-s, s, -s]),   # 3: 좌상단 앞
            np.array([-s, -s, s]),   # 4: 좌하단 뒤
            np.array([s, -s, s]),    # 5: 우하단 뒤
            np.array([s, s, s]),     # 6: 우상단 뒤
            np.array([-s, s, s])     # 7: 좌상단 뒤
        ]
        
        # 6개의 면 (각 면은 4개의 정점으로 구성)
        self.faces = [
            (0, 1, 2, 3),  # 앞면
            (4, 5, 6, 7),  # 뒷면
            (0, 4, 7, 3),  # 왼쪽면
            (1, 5, 6, 2),  # 오른쪽면
            (0, 1, 5, 4),  # 아래면
            (3, 2, 6, 7)   # 위면
        ]
        
    def create_wedge(self, size):
        """차량용 웨지 형태 메시 생성 (단순화된 차 모양)"""
        w, h, l = size  # 가로, 세로, 길이
        w2, h2, l2 = w/2, h/2, l/2
        
        # 6개의 정점 (아래 4개, 위 앞 1개, 위 뒤 1개)
        self.vertices = [
            np.array([-w2, -h2, -l2]),  # 0: 좌하단 앞
            np.array([w2, -h2, -l2]),   # 1: 우하단 앞
            np.array([w2, -h2, l2]),    # 2: 우하단 뒤
            np.array([-w2, -h2, l2]),   # 3: 좌하단 뒤
            np.array([0, h2, -l2*0.7]),  # 4: 상단 앞쪽 (뾰족하게)
            np.array([0, h2, l2])        # 5: 상단 뒤쪽
        ]
        
        # 5개의 면
        self.faces = [
            (0, 1, 4),         # 앞면 (삼각형)
            (1, 2, 5, 4),      # 오른쪽면
            (2, 3, 5),         # 뒷면 (삼각형)
            (3, 0, 4, 5),      # 왼쪽면
            (0, 1, 2, 3)       # 아래면
        ]
    
    def translate(self, position):
        """평행 이동된 정점 목록 반환"""
        return [v + position for v in self.vertices]
    
    def rotate_y(self, angle):
        """Y축 기준 회전된 정점 목록 반환"""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        return [np.dot(rotation_matrix, v) for v in self.vertices]
    
    def transform(self, position, rotation):
        """위치와 회전 적용된 정점 목록 반환"""
        # 먼저 회전
        rotated = self.rotate_y(rotation)
        # 이후 평행이동
        return [v + position for v in rotated]
    
    def draw(self, surface, position, rotation, color=None):
        """메시를 화면에 그리기"""
        if color is None:
            color = self.color
        
        # 변환 적용 (회전 후 이동)
        transformed_vertices = self.transform(position, rotation)
        
        # 화면 좌표로 변환
        screen_vertices = []
        for v in transformed_vertices:
            screen_point = world_to_screen(v)
            if screen_point:
                screen_vertices.append(screen_point)
            else:
                screen_vertices.append(None)
        
        # 면별로 그리기
        for face in self.faces:
            # 면의 모든 정점이 화면에 보이는지 확인
            points = []
            skip_face = False
            
            for idx in face:
                if screen_vertices[idx] is None:
                    skip_face = True
                    break
                points.append(screen_vertices[idx])
            
            if skip_face:
                continue
            
            # 간단한 백페이스 컬링 (뒷면 제거)
            if len(points) >= 3:
                # 화면상의 벡터 계산
                v1 = (points[1][0] - points[0][0], points[1][1] - points[0][1])
                v2 = (points[2][0] - points[0][0], points[2][1] - points[0][1])
                # 외적으로 방향 확인 (2D에서는 z 성분만)
                z_component = v1[0] * v2[1] - v1[1] * v2[0]
                if z_component <= 0:  # 시계 방향이면 뒷면
                    continue
            
            # Z-buffer 평균값 계산 (깊이)
            avg_z = sum(p[2] for p in points) / len(points)
            
            # 화면 좌표만 추출 (z 제외)
            screen_points = [(p[0], p[1]) for p in points]
            
            # 폴리곤 그리기
            pygame.draw.polygon(surface, color, screen_points)
            # 윤곽선 그리기
            pygame.draw.polygon(surface, BLACK, screen_points, 1)

# --- 트랙 클래스 ---
class Track:
    def __init__(self):
        self.points_3d = []  # 트랙 중심선 점들
        self.segments = []   # 트랙 세그먼트 (폴리곤)
        self.segment_lengths = []  # 각 세그먼트 길이
        self.total_length = 0      # 총 트랙 길이
        self.width = 8.0           # 트랙 폭
        
        self.create_track()
    
    def create_track(self):
        """트랙 생성 (향상된 버전: 높이 변화 추가)"""
        # 기본 타원형 경로 좌표
        segments = 120
        track_radius_x = 50
        track_radius_y = 30
        
        # 중심선 점들 생성
        center_points = []
        for i in range(segments):
            angle = (math.pi * 2 * i) / segments
            x = math.cos(angle) * track_radius_x
            y = math.sin(angle) * track_radius_y
            
            # 높이 변화 추가 (언덕과 골짜기)
            z = 0
            # 첫 번째 언덕 (30% ~ 40% 구간)
            if 0.3 <= i/segments <= 0.4:
                z = 5 * math.sin((i/segments - 0.3) * math.pi / 0.1)
            # 두 번째 언덕 (60% ~ 80% 구간)
            elif 0.6 <= i/segments <= 0.8:
                z = 8 * math.sin((i/segments - 0.6) * math.pi / 0.2)
            
            center_points.append(np.array([x, z, y]))  # 주의: y, z 축 변환
        
        self.points_3d = center_points
        
        # 세그먼트 길이 계산
        self.segment_lengths = []
        self.total_length = 0
        for i in range(segments):
            p1 = center_points[i]
            p2 = center_points[(i + 1) % segments]
            length = np.linalg.norm(p2 - p1)
            self.segment_lengths.append(length)
            self.total_length += length
        
        # 트랙 폴리곤 생성 (양쪽 가장자리 포함)
        self.segments = []
        for i in range(segments):
            p1 = center_points[i]
            p2 = center_points[(i + 1) % segments]
            
            # 트랙 방향 벡터
            direction = p2 - p1
            direction = normalize(direction)
            
            # 수직 벡터 (트랙 왼쪽/오른쪽)
            # y축이 높이이므로, xz 평면에서 수직이어야 함
            perpendicular = np.array([-direction[2], 0, direction[0]])
            perpendicular = normalize(perpendicular)
            
            # 트랙 가장자리 포인트
            left1 = p1 + perpendicular * self.width/2
            right1 = p1 - perpendicular * self.width/2
            left2 = p2 + perpendicular * self.width/2
            right2 = p2 - perpendicular * self.width/2
            
            # 세그먼트 저장 (사각형 형태)
            self.segments.append((left1, right1, right2, left2))
            
            # 경사진 커브 (뱅크 효과) 추가 - 곡률이 높은 곳에서
            # 첫 번째 급커브 (25% 지점)
            if 0.2 <= i/segments <= 0.3:
                # 바깥쪽 가장자리를 높이고 안쪽 가장자리는 낮춤
                bank_factor = 4 * math.sin((i/segments - 0.2) * math.pi / 0.1)
                left1[1] += bank_factor   # 바깥쪽 높이기
                left2[1] += bank_factor
            # 두 번째 급커브 (75% 지점)
            elif 0.7 <= i/segments <= 0.8:
                bank_factor = 4 * math.sin((i/segments - 0.7) * math.pi / 0.1)
                right1[1] += bank_factor  # 반대 방향이라 안쪽이 바뀜
                right2[1] += bank_factor
    
    def get_position_on_track(self, segment_idx, progress):
        """트랙 위의 특정 위치를 계산"""
        # 현재 세그먼트의 시작과 끝 점
        p1 = self.points_3d[segment_idx]
        p2 = self.points_3d[(segment_idx + 1) % len(self.points_3d)]
        
        # 세그먼트 내 위치 계산 (선형 보간)
        t = progress / self.segment_lengths[segment_idx] if self.segment_lengths[segment_idx] > 0 else 0
        position = p1 + (p2 - p1) * t
        
        # 방향 벡터 계산
        direction = normalize(p2 - p1) if np.linalg.norm(p2 - p1) > 0 else np.array([1, 0, 0])
        
        return position, direction
    
    def draw(self, surface):
        """트랙 그리기"""
        # 트랙 세그먼트 그리기
        for segment in self.segments:
            points = []
            valid_segment = True
            
            # 각 점을 화면 좌표로 변환
            for point in segment:
                screen_point = world_to_screen(point)
                if screen_point:
                    points.append((screen_point[0], screen_point[1]))
                else:
                    valid_segment = False
                    break
            
            if valid_segment:
                # 아스팔트 색
                pygame.draw.polygon(surface, TRACK_COLOR, points)
                # 트랙 경계선
                pygame.draw.polygon(surface, WHITE, points, 1)
        
        # 결승선 그리기
        finish_line_start = self.points_3d[0] + np.array([-self.width/2, 0.1, 0])
        finish_line_end = self.points_3d[0] + np.array([self.width/2, 0.1, 0])
        
        start_screen = world_to_screen(finish_line_start)
        end_screen = world_to_screen(finish_line_end)
        
        if start_screen and end_screen:
            pygame.draw.line(surface, WHITE, 
                            (start_screen[0], start_screen[1]), 
                            (end_screen[0], end_screen[1]), 3)

# --- 차량 클래스 개선 ---
class Car:
    def __init__(self, car_id, name, color, start_pos_offset):
        self.id = car_id
        self.name = name
        self.color = color
        self.size = (2.0, 1.0, 4.0)  # 차량 크기 (가로, 높이, 길이)
        
        # 메시 생성
        self.mesh = Mesh()
        self.mesh.create_wedge(self.size)
        self.mesh.color = color
        
        # 위치 및 경로 관련
        self.track_index = 0
        self.segment_progress = 0.0
        self.position_3d = np.copy(track.points_3d[0])
        self.position_3d[0] += start_pos_offset[0]  # 옆으로 위치 조정
        self.position_3d[2] += start_pos_offset[1]  # 앞뒤로 위치 조정
        self.direction = np.array([0.0, 0.0, 1.0])  # 초기 방향
        self.rotation = 0.0  # Y축 기준 회전 각도 (라디안)
        self.lap = 0
        self.finished = False
        self.finish_time = 0
        
        # 성능 관련
        self.speed = 0.0
        self.base_max_speed = 20.0 + random.uniform(-4.0, 8.0)
        self.effective_max_speed = self.base_max_speed
        self.acceleration = 5.0 + random.uniform(-1.0, 3.0)
        self.handling = 1.0 + random.uniform(-0.2, 0.3)  # 핸들링 능력 (차량별 차이)
        
        # 물리 관련
        self.mass = 1000.0  # kg
        self.drag = 0.3     # 공기 저항
        self.rolling_resistance = 0.1  # 지면 마찰
        
        # 이벤트 관련
        self.speed_modifier_duration = 0.0
        self.speed_modifier_amount = 0.0
        self.is_leading = False
        self.distance_behind_leader = 0.0
    
    def get_total_distance(self):
        """현재까지 이동한 총 경로 거리 계산"""
        dist = self.lap * track.total_length
        for i in range(self.track_index):
            dist += track.segment_lengths[i]
        dist += self.segment_progress
        return dist
    
    def update(self, dt, leader_car):
        if self.finished:
            self.speed *= 0.95  # 완주 후 천천히 감속
            # 마지막 방향대로 계속 이동
            self.position_3d += self.direction * self.speed * dt
            return
        
        # --- 이벤트 업데이트 ---
        self.speed_modifier_duration = max(0, self.speed_modifier_duration - dt)
        if self.speed_modifier_duration <= 0:
            self.speed_modifier_amount = 0
            # 이벤트 발생 로직
            event_chance = 0.015 * (60 * dt)
            
            # 선두/후미 로직
            if self.is_leading:
                event_chance *= 0.7  # 선두는 이벤트 확률 감소
            elif self.distance_behind_leader > track.total_length * 0.3:
                event_chance *= 1.5  # 뒤처지면 이벤트 확률 증가
            
            if random.random() < event_chance:
                is_boost = random.random() < 0.6  # 60% 확률로 부스트
                if is_boost:
                    self.speed_modifier_amount = self.base_max_speed * random.uniform(0.4, 0.8)
                    self.speed_modifier_duration = random.uniform(1.5, 3.5)
                else:
                    self.speed_modifier_amount = -self.base_max_speed * random.uniform(0.3, 0.6)
                    self.speed_modifier_duration = random.uniform(1.0, 2.0)
        
        # --- 물리 업데이트 ---
        # 유효 최고 속도 계산
        self.effective_max_speed = max(self.base_max_speed * 0.4, 
                                      self.base_max_speed + self.speed_modifier_amount)
        
        # 힘 계산
        traction_force = self.acceleration * 1000.0  # 엔진 힘
        drag_force = self.drag * self.speed * self.speed  # 공기 저항
        rolling_force = self.rolling_resistance * self.speed  # 지면 마찰
        
        # 총 힘
        if self.speed < self.effective_max_speed:
            total_force = traction_force - drag_force - rolling_force
        else:
            total_force = -drag_force - rolling_force
        
        # 가속도 계산 (F = ma)
        acceleration = total_force / self.mass
        
        # 속도 업데이트
        self.speed += acceleration * dt
        self.speed = max(0, self.speed)
        
        # --- 트랙 경로 이동 ---
        distance_to_move = self.speed * dt
        
        while distance_to_move > 0 and not self.finished:
            current_segment_len = track.segment_lengths[self.track_index]
            remaining_in_segment = current_segment_len - self.segment_progress
            
            if distance_to_move >= remaining_in_segment:
                distance_to_move -= remaining_in_segment
                self.segment_progress = 0
                self.track_index = (self.track_index + 1) % len(track.points_3d)
                
                # 랩 완료 체크
                if self.track_index == 0:
                    self.lap += 1
                    print(f"Car {self.id} completed lap {self.lap}")
                    if self.lap >= LAPS_TO_WIN:
                        self.finished = True
                        self.finish_time = pygame.time.get_ticks() / 1000.0
                        break
            else:
                self.segment_progress += distance_to_move
                distance_to_move = 0
        
        # 현재 트랙 위치 및 방향 계산
        track_position, track_direction = track.get_position_on_track(
            self.track_index, self.segment_progress)
        
        # 트랙 위치로 이동
        self.position_3d = track_position
        
        # 커브 핸들링: 방향을 부드럽게 조정 (핸들링 능력에 따라 다름)
        self.direction = self.direction * (1 - self.handling * 0.1) + track_direction * (self.handling * 0.1)
        self.direction = normalize(self.direction)
        
        # 회전 각도 계산 (Y축 기준 회전)
        self.rotation = math.atan2(self.direction[2], self.direction[0])
    
    def draw(self, surface):
        # 메시 그리기 (위치와 회전 적용)
        self.mesh.draw(surface, self.position_3d, self.rotation, self.color)
        
        # 부스트/슬로우 효과 표시 (파티클 효과 등으로 개선 가능)
        if self.speed_modifier_amount > 0:
            # 부스트 효과 (간단한 불꽃 파티클)
            flame_pos = self.position_3d - self.direction * 2.0
            flame_pos[1] += 0.5  # 약간 위로
            flame_screen = world_to_screen(flame_pos)
            if flame_screen:
                pygame.draw.circle(surface, YELLOW, (flame_screen[0], flame_screen[1]), 5)
        elif self.speed_modifier_amount < 0:
            # 슬로우 효과 (간단한 연기 파티클)
            smoke_pos = self.position_3d - self.direction * 1.5
            smoke_pos[1] += 1.0  # 위로
            smoke_screen = world_to_screen(smoke_pos)
            if smoke_screen:
                pygame.draw.circle(surface, GRAY, (smoke_screen[0], smoke_screen[1]), 8)

# --- 카메라 컨트롤러 ---
class CameraController:
    def __init__(self):
        self.target_car = None
        self.mode = "chase"  # chase, overhead, first_person 등
        self.transition_speed = 0.1
        
    def set_target(self, car):
        self.target_car = car
    
    def toggle_mode(self):
        """카메라 모드 전환"""
        if self.mode == "chase":
            self.mode = "overhead"
        elif self.mode == "overhead":
            self.mode = "first_person"
        else:
            self.mode = "chase"
        print(f"Camera mode: {self.mode}")
    
    def update(self, dt):
        if not self.target_car:
            return
        
        # 목표 차량 위치와 방향
        car_pos = self.target_car.position_3d
        car_dir = self.target_car.direction
        
        if self.mode == "chase":
            # 추적 카메라 (차량 뒤에서 따라감)
            target_pos = car_pos - car_dir * 12.0 + np.array([0, 6.0, 0])
            target_look = car_pos + car_dir * 5.0
        elif self.mode == "overhead":
            # 오버헤드 뷰 (위에서 내려다봄)
            target_pos = car_pos + np.array([0, 30.0, 0])
            target_look = car_pos
        elif self.mode == "first_person":
            # 일인칭 뷰 (차량 안에서 바라봄)
            target_pos = car_pos + np.array([0, 1.2, 0]) # 약간 높은 위치 (운전석)
            target_look = car_pos + car_dir * 10.0
        
        # 카메라 위치와 시선 부드럽게 이동
        camera['position'] = camera['position'] * (1 - self.transition_speed) + target_pos * self.transition_speed
        camera['target'] = camera['target'] * (1 - self.transition_speed) + target_look * self.transition_speed

# --- 게임 상태 변수 ---
game_state = "menu"  # menu, racing, finish
cars = []
finished_cars_ordered = []
start_time = 0
winner = None
player_count = 4
selected_car_index = 0

# --- 트랙 생성 ---
track = Track()

# --- 카메라 컨트롤러 ---
camera_controller = CameraController()

# --- 메뉴 UI 요소 ---
def draw_menu():
    """메뉴 화면 그리기"""
    screen.fill(BLACK)
    
    # 타이틀
    title = font.render("3D Racing Game", True, WHITE)
    title_rect = title.get_rect(center=(SCREEN_WIDTH/2, 100))
    screen.blit(title, title_rect)
    
    # 플레이어 수 선택
    players_text = font.render(f"Players: {player_count}", True, WHITE)
    players_rect = players_text.get_rect(center=(SCREEN_WIDTH/2, 200))
    screen.blit(players_text, players_rect)
    
    # 플레이어 수 변경 버튼
    less_text = font.render("<", True, WHITE)
    less_rect = less_text.get_rect(center=(SCREEN_WIDTH/2 - 100, 200))
    screen.blit(less_text, less_rect)
    
    more_text = font.render(">", True, WHITE)
    more_rect = more_text.get_rect(center=(SCREEN_WIDTH/2 + 100, 200))
    screen.blit(more_text, more_rect)
    
    # 차량 선택 (플레이어 차량)
    select_text = font.render("Your Car:", True, WHITE)
    select_rect = select_text.get_rect(center=(SCREEN_WIDTH/2, 300))
    screen.blit(select_text, select_rect)
    
    # 차량 색상 표시
    car_colors = [RED, BLUE, GREEN, YELLOW, (255, 165, 0), (128, 0, 128)]
    car_color = car_colors[selected_car_index % len(car_colors)]
    pygame.draw.rect(screen, car_color, pygame.Rect(SCREEN_WIDTH/2 - 25, 340, 50, 30))
    
    # 차량 선택 변경 버튼
    car_less = font.render("<", True, WHITE)
    car_less_rect = car_less.get_rect(center=(SCREEN_WIDTH/2 - 50, 355))
    screen.blit(car_less, car_less_rect)
    
    car_more = font.render(">", True, WHITE)
    car_more_rect = car_more.get_rect(center=(SCREEN_WIDTH/2 + 50, 355))
    screen.blit(car_more, car_more_rect)
    
    # 시작 버튼
    start_text = font.render("Start Race!", True, WHITE)
    start_rect = start_text.get_rect(center=(SCREEN_WIDTH/2, 450))
    pygame.draw.rect(screen, GREEN, start_rect.inflate(20, 10))
    screen.blit(start_text, start_rect)
    
    # 설명
    instructions = small_font.render("C: Change camera view  |  ESC: Quit", True, WHITE)
    instructions_rect = instructions.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT - 50))
    screen.blit(instructions, instructions_rect)
    
    return {
        'less_players': less_rect,
        'more_players': more_rect,
        'car_less': car_less_rect,
        'car_more': car_more_rect,
        'start': start_rect
    }

def start_race():
    """레이스 시작"""
    global game_state, cars, finished_cars_ordered, start_time
    
    game_state = "racing"
    cars = []
    finished_cars_ordered = []
    
    # 차량 색상
    car_colors = [RED, BLUE, GREEN, YELLOW, (255, 165, 0), (128, 0, 128)]
    
    # 차량 생성
    for i in range(player_count):
        # 이름 생성
        if i == 0:
            name = "Player"
            color = car_colors[selected_car_index % len(car_colors)]
        else:
            name = f"AI {i}"
            color = car_colors[(selected_car_index + i) % len(car_colors)]
        
        # 시작 위치 오프셋 (그리드 형태로 배치)
        row = i // 2
        col = i % 2
        offset_x = (col * 2 - 0.5) * 3.0  # 좌우 배치
        offset_z = -row * 6.0  # 앞뒤 배치 (음수 값이 뒤쪽)
        
        car = Car(i + 1, name, color, (offset_x, offset_z))
        cars.append(car)
    
    # 플레이어 차량을 카메라 타겟으로 설정
    if cars:
        camera_controller.set_target(cars[0])
    
    start_time = pygame.time.get_ticks()

# --- 게임 루프 ---
running = True
while running:
    # --- 이벤트 처리 ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            
            elif event.key == pygame.K_c:
                # 카메라 모드 변경
                camera_controller.toggle_mode()
            
            elif event.key == pygame.K_RETURN and game_state == "menu":
                start_race()
        
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # 마우스 왼쪽 버튼 클릭
            mouse_pos = pygame.mouse.get_pos()
            
            if game_state == "menu":
                ui_rects = draw_menu()
                
                if ui_rects['less_players'].collidepoint(mouse_pos):
                    player_count = max(2, player_count - 1)
                
                elif ui_rects['more_players'].collidepoint(mouse_pos):
                    player_count = min(6, player_count + 1)
                
                elif ui_rects['car_less'].collidepoint(mouse_pos):
                    selected_car_index = (selected_car_index - 1) % 6
                
                elif ui_rects['car_more'].collidepoint(mouse_pos):
                    selected_car_index = (selected_car_index + 1) % 6
                
                elif ui_rects['start'].collidepoint(mouse_pos):
                    start_race()
    
    # --- 시간 처리 ---
    dt = clock.tick(FPS) / 1000.0
    
    # --- 게임 로직 업데이트 ---
    if game_state == "racing":
        # 카메라 업데이트
        camera_controller.update(dt)
        
        # Z-버퍼 초기화
        z_buffer.clear()
        
        # 선두 차량 찾기
        leading_car = None
        max_dist = -1
        active_cars = [car for car in cars if not car.finished]
        
        if active_cars:
            for car in active_cars:
                dist = car.get_total_distance()
                if dist > max_dist:
                    max_dist = dist
                    leading_car = car
            
            if leading_car:
                leading_car.is_leading = True
                for car in active_cars:
                    if car != leading_car:
                        car.is_leading = False
                        car.distance_behind_leader = max(0, max_dist - car.get_total_distance())
                    else:
                        car.distance_behind_leader = 0
        
        # 각 차량 업데이트
        for car in cars:
            was_finished = car.finished
            car.update(dt, leading_car)
            if car.finished and not was_finished:
                if car not in finished_cars_ordered:
                    finished_cars_ordered.append(car)
                    print(f"Car {car.id} finished in position {len(finished_cars_ordered)}")
        
        # 게임 종료 조건 확인
        if len(finished_cars_ordered) == len(cars):
            game_state = "finish"
            winner = finished_cars_ordered[0]
            print("Race Finished!")
    
    # --- 렌더링 ---
    if game_state == "menu":
        # 메뉴 그리기
        draw_menu()
    
    else:  # racing 또는 finish 상태
        # 배경
        screen.fill(GRASS_COLOR)
        
        # 하늘 그리기 (간단한 구현)
        sky_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT//2)
        pygame.draw.rect(screen, (135, 206, 235), sky_rect)  # 하늘색
        
        # 트랙 그리기
        track.draw(screen)
        
        # 차량 그리기 (깊이 정렬)
        cars_to_draw = sorted(cars, key=lambda c: -np.linalg.norm(c.position_3d - camera['position']))
        for car in cars_to_draw:
            car.draw(screen)
        
        # UI 그리기
        if game_state == "racing" or game_state == "finish":
            # 랩 카운터
            display_lap = 0
            if leading_car:
                display_lap = min(leading_car.lap + 1, LAPS_TO_WIN)
            elif finished_cars_ordered:
                display_lap = LAPS_TO_WIN
            else:
                display_lap = 1 if cars else 0
            
            lap_text = font.render(f"Lap: {display_lap} / {LAPS_TO_WIN}", True, WHITE)
            lap_rect = lap_text.get_rect(topleft=(SCREEN_WIDTH - 150, 10))
            pygame.draw.rect(screen, BLACK, lap_rect.inflate(10, 5))
            screen.blit(lap_text, lap_rect)
            
            # 순위 정보
            if finished_cars_ordered:
                y_offset = 50
                rank_text_header = small_font.render("Ranking:", True, WHITE)
                rank_header_rect = rank_text_header.get_rect(topleft=(SCREEN_WIDTH - 150, y_offset))
                pygame.draw.rect(screen, BLACK, rank_header_rect.inflate(10, 5))
                screen.blit(rank_text_header, rank_header_rect)
                
                y_offset += 25
                for i, car in enumerate(finished_cars_ordered):
                    medal = ""
                    if i == 0: medal = "🥇"
                    elif i == 1: medal = "🥈"
                    elif i == 2: medal = "🥉"
                    
                    rank_text = small_font.render(f"{medal} Car {car.id} ({car.name})", True, WHITE)
                    rank_rect = rank_text.get_rect(topleft=(SCREEN_WIDTH - 150, y_offset))
                    pygame.draw.rect(screen, BLACK, rank_rect.inflate(10, 5))
                    screen.blit(rank_text, rank_rect)
                    y_offset += 20
        
        # 완주 화면
        if game_state == "finish" and winner:
            win_text = font.render(f"Winner: Car {winner.id} ({winner.name})!", True, YELLOW)
            win_rect = win_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
            pygame.draw.rect(screen, BLACK, win_rect.inflate(30, 15))
            screen.blit(win_text, win_rect)
            
            restart_text = small_font.render("Press ENTER to return to menu", True, WHITE)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 40))
            screen.blit(restart_text, restart_rect)
            
            # Enter 키 입력 시 메뉴로 돌아가기
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RETURN]:
                game_state = "menu"
    
    # --- 화면 업데이트 ---
    pygame.display.flip()

# --- 게임 종료 ---
pygame.quit()