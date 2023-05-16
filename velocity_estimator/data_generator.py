import numpy as np
import pickle
from fire import Fire
from tqdm import tqdm

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
                 angle_xy_range=(-20, 20),
                 angle_y_range=(-20, 20),
                 slow_motion_coefs=[1, 2, 4, 8],
                 ball_size_range=(7/640, 1/8),
                 velocity_range=(15, 45),
                 avg_hits=3.,
                 pos_std_err=0.003,
                 d_std_err=0.1,
                 random_object_prob=0.15,
                 no_object_prob=0.5,
                 hit_max_z_angle=15,
                 field_size=30,
                 padding_size=0.25):
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.angle_xy_range = angle_xy_range
        self.angle_y_range = angle_y_range
        self.slow_motion_coefs = slow_motion_coefs
        self.ball_size_range = ball_size_range
        self.velocity_range = velocity_range
        self.avg_hits = avg_hits
        self.pos_std_err = pos_std_err
        self.d_std_err = d_std_err
        self.random_object_prob = random_object_prob
        self.no_object_prob = no_object_prob
        self.field_size = field_size
        self.hit_max_z_angle = hit_max_z_angle
        self.min_hit_distance = self.get_position_z(ball_size_range[1]) + np.sin(np.pi/180*hit_max_z_angle)*velocity_range[1]*2/DEFAULT_FRAMES_PER_SEC
        self.max_hit_distance = self.get_position_z(ball_size_range[0]) - np.sin(np.pi/180*hit_max_z_angle)*velocity_range[1]*2/DEFAULT_FRAMES_PER_SEC
        self.padding_size = padding_size
        self.params = self.__dict__.copy()
        print("Initalization params:\n", ", ".join([f"{k}={v}" for k, v in self.params.items()]))
    
    @property
    def velocity_norm(self):
        return np.linalg.norm(self.velocity, axis=-1)

    def draw_parameters(self, generate_start_position=True):
        if generate_start_position:
            self.diff_time = 1/(DEFAULT_FRAMES_PER_SEC*np.random.choice(self.slow_motion_coefs, size=(self.n_samples, 1)))

            self.angle_xy = np.random.uniform(*self.angle_xy_range, size=self.n_samples)
            self.angle_y = np.random.uniform(*self.angle_y_range, size=self.n_samples)
            self.start_position = np.concatenate([np.random.uniform(-2, 2, size=(self.n_samples, 2)),
                                                log_rand(self.min_hit_distance, self.max_hit_distance, size=(self.n_samples, 1))], axis=-1)

            self.start_velocity_norm = log_rand(*self.velocity_range, size=(self.n_samples, 1))
            hit_angle_xy = np.random.uniform(0, 2*np.pi, size=(self.n_samples, 1))
            hit_angle_z = np.random.uniform(-np.pi*self.hit_max_z_angle/180, np.pi*self.hit_max_z_angle/180, size=(self.n_samples, 1))
            self.start_velocity = np.concatenate([self.start_velocity_norm*np.cos(hit_angle_z)*np.cos(hit_angle_xy),
                                                self.start_velocity_norm*np.cos(hit_angle_z)*np.sin(hit_angle_xy),
                                                self.start_velocity_norm*np.sin(hit_angle_z)], axis=-1)
        else:
            self.start_velocity_norm = np.random.uniform(low=0, high=0.5*self.start_velocity_norm, size=(self.n_samples, 1))
            hit_angle_xy = np.random.uniform(0, 2*np.pi, size=(self.n_samples, 1))
            hit_angle_z = np.random.uniform(-np.pi*self.hit_max_z_angle/180, np.pi*self.hit_max_z_angle/180, size=(self.n_samples, 1))
            self.start_velocity = np.concatenate([self.start_velocity_norm*np.cos(hit_angle_z)*np.cos(hit_angle_xy),
                                                self.start_velocity_norm*np.cos(hit_angle_z)*np.sin(hit_angle_xy),
                                                self.start_velocity_norm*np.sin(hit_angle_z)], axis=-1)
        
        velocity_after_hit_norm = np.random.uniform(low=0, high=0.5*np.expand_dims(self.start_velocity_norm, axis=1), size=(self.n_samples, self.n_frames, 1))
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
    
    def generate_trajectories(self, generate_start_position=True):
        self.draw_parameters(generate_start_position)
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
        return 4/np.pi*np.arctan(VOLLEYBALL_DIAMETER/positions[:, :, 2:])
    
    def get_position_z(self, diameter):
        return VOLLEYBALL_DIAMETER/np.tan(np.pi/4*diameter)
    
    def concat_trajectories(self, positions, velocities, backward_positions, backward_velocities):
        backward_positions = backward_positions[:, 1:][:, ::-1]
        backward_velocities = backward_velocities[:, 1:][:, ::-1]

        positions = np.concatenate([backward_positions, positions], axis=1)
        velocities = np.concatenate([backward_velocities, velocities], axis=1)

        window_start = np.random.randint(0, self.n_frames-1, size=self.n_samples)
        window = np.linspace(window_start, window_start+self.n_frames-1, num=self.n_frames, dtype=int)
        window = np.expand_dims(np.transpose(window, (1, 0)), axis=-1)

        positions = np.take_along_axis(positions, window, axis=1)
        velocities = np.take_along_axis(velocities, window, axis=1)

        return positions, velocities

    def normalize(self, positions):
        """
        Normalizes real 3D position coordinates (in meters) to image coordinates (in pixels).
        """
        image_positions = positions[:, :, :2]/self.field_size+0.5
        image_diameter = self.get_diameter(positions)

        return image_positions, image_diameter

    def add_noise(self, image_positions, image_diameter):
        pos_noise = np.random.normal(0., self.pos_std_err, size=image_positions.shape)
        image_positions = image_positions + pos_noise

        d_noise = image_diameter*np.random.lognormal(0., np.log(1+self.d_std_err), size=image_diameter.shape)
        image_diameter = image_diameter + d_noise
        
        should_be_random = np.random.uniform(0, 1, size=image_diameter.shape) < self.random_object_prob
        random_positions = np.random.uniform(0, 1, size=image_positions.shape)
        random_d = log_rand(*self.ball_size_range, size=image_diameter.shape)
        image_positions = np.where(should_be_random == 0,
                                image_positions,
                                random_positions)
        image_diameter = np.where(should_be_random == 0,
                                image_diameter,
                                random_d)
        
        should_be_no_object = np.random.uniform(0, 1, size=image_diameter.shape) < self.no_object_prob
        image_positions = np.where(should_be_no_object == 0,
                                image_positions,
                                -1)
        image_diameter = np.where(should_be_no_object == 0,
                                image_diameter,
                                -1)
        
        return image_positions, image_diameter
    
    def clip(self, image_positions, image_diameter, velocities):
        image_positions = np.where(np.any((image_positions < 0) | (image_positions > 1), axis=-1, keepdims=True),
                                   -1,
                                   image_positions)
        image_diameter = np.where(image_diameter < VOLLEYBALL_DIAMETER/self.field_size,
                                  -1,
                                  image_diameter)
        velocities = np.where(np.any(image_positions == -1, axis=-1, keepdims=True),
                              -1,
                              velocities)
        velocities = np.where(image_diameter == -1,
                              -1,
                              velocities)

        return image_positions, image_diameter, velocities
    
    def random_padding(self, image_positions, image_diameter, velocities):
        mask = np.ones((self.n_samples, self.n_frames, 1), dtype=int)
        padding_size = np.random.randint(0, int(self.n_frames*self.padding_size), size=self.n_samples)
        padding_mode = np.random.choice(['left', 'right'], size=self.n_samples)
        for i in range(self.n_samples):
            if padding_mode[i] == 'left':
                mask[i, :padding_size[i]] = 0
            else:
                mask[i, -padding_size[i]:] = 0

        image_positions = np.where(mask == 0,
                                   -1,
                                   image_positions)
        image_diameter = np.where(mask == 0,
                                    -1,
                                    image_diameter)
        velocities = np.where(mask == 0,
                                -1,
                                velocities)
        
        return image_positions, image_diameter, velocities
    
    def __call__(self, add_noise=True, random_padding=True, clip=True):
        positions, velocities = self.generate_trajectories(generate_start_position=True)
        self.diff_time = -self.diff_time
        backward_positions, backward_velocities = self.generate_trajectories(generate_start_position=False)
        positions, velocities = self.concat_trajectories(positions, velocities, backward_positions, backward_velocities)
        image_positions, image_diameter = self.normalize(positions)
        if add_noise:
            image_positions, image_diameter = self.add_noise(image_positions, image_diameter)
        if clip:
            image_positions, image_diameter, velocities = self.clip(image_positions, image_diameter, velocities)
        if random_padding:
            image_positions, image_diameter, velocities = self.random_padding(image_positions, image_diameter, velocities)

        return image_positions, image_diameter, velocities
    
def generate(data_path, n_samples, n_frames, batch_size=1000, **kwargs):
    image_positions = np.zeros((n_samples, n_frames, 2))
    image_diameter = np.zeros((n_samples, n_frames, 1))
    velocities = np.zeros((n_samples, n_frames, 3))
    generator = TrajectoriesGenerator(min(n_samples, batch_size), n_frames, **kwargs)
    for i in tqdm(range(0, n_samples, batch_size), unit_scale=batch_size):
        image_positions[i:i+batch_size], image_diameter[i:i+batch_size], velocities[i:i+batch_size] = generator()

    pickle.dump((image_positions, image_diameter, velocities), open(data_path, 'wb'))

if __name__ == "__main__":
    Fire(generate)