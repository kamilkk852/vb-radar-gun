import numpy as np

G = 9.81
VOLLEYBALL_DIAMETER = 0.21
DEFAULT_FRAMES_PER_SEC = 30

def log_rand(low, high, **kwargs):
    log_rand = np.random.uniform(np.log(low), np.log(high), **kwargs)
    return np.exp(log_rand)

class TrajectoriesGenerator:
    def __init__(self,
                 n_samples,
                 n_frames,
                 angle_xy_range=(0, 0),
                 angle_y_range=(0, 0),
                 slow_motion_coefs=[1, 2, 4, 8],
                 ball_size_range=(7/640, 1/8),
                 velocity_range=(15, 45),
                 avg_hits=3.,
                 pos_std_err=0.002,
                 d_std_err=0.1,
                 random_object_prob=0.15,
                 random_no_object_prob=0.5,
                 hit_max_z_angle=45,
                 min_hit_distance=3,
                 field_size=30):
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.diff_time = 1/(DEFAULT_FRAMES_PER_SEC*np.random.choice(slow_motion_coefs, size=(n_samples, 1)))
        self.angle_xy_range = angle_xy_range
        self.angle_y_range = angle_y_range
        self.slow_motion_coefs = slow_motion_coefs
        self.ball_size_range = ball_size_range
        self.velocity_range = velocity_range
        self.avg_hits = avg_hits
        self.pos_std_err = pos_std_err
        self.d_std_err = d_std_err
        self.random_object_prob = random_object_prob
        self.random_no_object_prob = random_no_object_prob
        self.field_size = field_size
        self.hit_max_z_angle = hit_max_z_angle
        self.min_hit_distance = min_hit_distance
    
    @property
    def velocity_norm(self):
        return np.linalg.norm(self.velocity, axis=-1)

    def draw_parameters(self):
        self.angle_xy = np.random.uniform(*self.angle_xy_range, size=self.n_samples)
        self.angle_y = np.random.uniform(*self.angle_y_range, size=self.n_samples)
        self.start_position = np.concatenate([np.random.uniform(-2, 2, size=(self.n_samples, 2)),
                                              np.random.uniform(self.min_hit_distance, 2*self.field_size, size=(self.n_samples, 1))], axis=-1)

        start_velocity_norm = log_rand(*self.velocity_range, size=(self.n_samples, 1))
        hit_angle_xy = np.random.uniform(0, 2*np.pi, size=(self.n_samples, 1))
        hit_angle_z = np.random.uniform(-np.pi*self.hit_max_z_angle/180, np.pi*self.hit_max_z_angle/180, size=(self.n_samples, 1))
        self.start_velocity = np.concatenate([start_velocity_norm*np.cos(hit_angle_z)*np.cos(hit_angle_xy),
                                              start_velocity_norm*np.cos(hit_angle_z)*np.sin(hit_angle_xy),
                                              start_velocity_norm*np.sin(hit_angle_z)], axis=-1)
        
        velocity_after_hit_norm = np.random.uniform(low=0, high=0.95*np.expand_dims(start_velocity_norm, axis=1), size=(self.n_samples, self.n_frames, 1))
        hit_angle_xy = np.random.uniform(0, 2*np.pi, size=(self.n_samples, self.n_frames, 1))
        hit_angle_z = np.random.uniform(-np.pi, np.pi, size=(self.n_samples, self.n_frames, 1))
        velocity_after_hit = np.concatenate([velocity_after_hit_norm*np.cos(hit_angle_z)*np.cos(hit_angle_xy),
                                             velocity_after_hit_norm*np.cos(hit_angle_z)*np.sin(hit_angle_xy),
                                             velocity_after_hit_norm*np.sin(hit_angle_z)], axis=-1)

        is_hit = np.random.uniform(0, 1, size=(self.n_samples, self.n_frames, 1)) < self.avg_hits/self.n_frames
        self.hits = np.concatenate([is_hit.astype(float), velocity_after_hit], axis=-1)
    
    def update_position(self, i):
        self.position[:, i] += self.velocity[:, i]*self.diff_time

    def add_gravitation(self, i):
        G_xy = G*np.cos(self.angle_xy*np.pi/180)
        G_z = G*np.sin(self.angle_xy*np.pi/180)
        G_y = G_xy*np.cos(self.angle_y*np.pi/180)
        G_x = G_xy*np.sin(self.angle_y*np.pi/180)
            
        a_vec = np.stack([G_x, G_y, G_z], axis=-1)
        
        self.velocity[:, i] = self.velocity[:, i]+a_vec*self.diff_time

    def add_air_resistance(self, i):
        Cx = 0.45
        m = 0.27
        d = VOLLEYBALL_DIAMETER
        A = np.pi*(d/2)**2
        rho = 1.2
        F = -0.5*rho*A*Cx*self.velocity[:, i]*np.expand_dims(self.velocity_norm[:, i], axis=-1)
        a = F/m
        
        self.velocity[:, i] = self.velocity[:, i] + a*self.diff_time

    def add_hits(self, i):
        self.velocity[:, i] = np.where(self.hits[:, i, :1] == 1, self.hits[:, i, 1:], self.velocity[:, i])
    
    def generate_trajectories(self):
        self.position = np.zeros((self.n_samples, self.n_frames, 3))
        self.velocity = np.zeros((self.n_samples, self.n_frames, 3))
        self.position[:, 0] = self.start_position.copy()
        self.velocity[:, 0] = self.start_velocity.copy()
        
        for i in range(1, self.n_frames):
            self.position[:, i] = self.position[:, i-1].copy()
            self.velocity[:, i] = self.velocity[:, i-1].copy()

            self.update_position(i)
            self.add_air_resistance(i)
            self.add_gravitation(i)
            self.add_hits(i)

        return self.position, self.velocity
    
    def get_diameter(self, positions):
        return np.pi/2*np.arctan(VOLLEYBALL_DIAMETER/positions[:, :, 2:])
    
    def normalize(self, positions):
        """
        Normalizes real 3D position coordinates (in meters) to image coordinates (in pixels).
        """
        image_positions = positions[:, :, :2]/self.field_size+0.5
        image_diameter = self.get_diameter(positions)

        return image_positions, image_diameter

    def add_noise(self, image_positions, image_diameter):
        pos_noise = np.random.normal(0., self.pos_noise_std, size=image_positions.shape)
        d_noise = image_diameter*np.random.lognormal(0., np.log(1+self.d_noise_std), size=positions[:, :, 2:].shape)
        noise = np.concatenate([pos_noise, d_noise], axis=-1)
        positions = positions + noise
        
        should_be_random = np.random.uniform(0, 1, size=positions[:, :, :1].shape) < self.random_object_prob
        random_poses = np.random.uniform(0, 1, size=image_positions.shape)
        random_d = log_rand(*self.d_range, size=positions[:, :, 2:].shape)
        random_values = np.concatenate([random_poses, random_d], axis=-1)
        positions = np.where(should_be_random == 0, positions, random_values)
        
        should_be_no_object = np.random.uniform(0, 1, size=positions[:, :, :1].shape) < self.no_object_prob
        positions = np.where(should_be_no_object == 0, positions, -1)
        
        return positions
    
    def clip(self, image_positions, image_diameter):
        image_positions = np.where(np.any((image_positions < 0) | (image_positions > 1), axis=-1, keepdims=True),
                                   -1,
                                   image_positions)
        image_diameter = np.where(image_diameter < VOLLEYBALL_DIAMETER/self.field_size,
                                  -1,
                                  image_diameter)

        
        return image_positions, image_diameter
    
    def __call__(self, add_noise=True, clip=True):
        self.draw_parameters()
        positions, velocities = self.generate_trajectories()
        image_positions, image_diameter = self.normalize(positions)
        # Not yet working
        # if add_noise:
        #     image_positions, image_diameter = self.add_noise(image_positions, image_diameter)
        if clip:
            image_positions, image_diameter = self.clip(image_positions, image_diameter)

        return image_positions, image_diameter, velocities
    