import mlx.core as mx
from numpy import angle


class Env:
    def __init__(
        self,
        m: float,
        l: float,
        g: float = 9.81,
        dt: float = 0.02,
        max_torque: float = 2.0,
        theta_threshold: float = mx.pi / 2,
        max_theta_dot: float = 10.0,
        init_theta_range: float = mx.pi,
        init_theta_dot_range: float = 1.0,
    ) -> None:
        self.m = mx.array(m)
        self.l = mx.array(l)
        self.g = mx.array(g)
        self.I = mx.array(m * (l**2))
        self.dt = mx.array(dt)
        self.max_torque = mx.array(max_torque)
        self.theta_threshold = mx.array(theta_threshold)
        self.max_theta_dot = mx.array(max_theta_dot)
        self.init_theta_range = mx.array(init_theta_range)
        self.init_theta_dot_range = mx.array(init_theta_dot_range)
        self.state: mx.array | None = None

    def observe(self) -> mx.array:
        if self.state is None:
            raise RuntimeError("call reset to set initial state")

        theta, theta_dot = self.state
        obs = mx.stack([mx.sin(theta), mx.cos(theta), theta_dot], axis=0)
        return mx.reshape(obs, (-1,))

    def reset(self, seed=None):
        if seed is not None:
            mx.random.seed(seed)

        theta = mx.random.uniform(-self.init_theta_range, self.init_theta_range)
        theta_dot = mx.random.uniform(-self.init_theta_dot_range, self.init_theta_dot_range)
        self.state = mx.array([theta, theta_dot])
        return self.observe()

    def step(self, action):
        assert self.state is not None
        u = mx.clip(action, -self.max_torque, self.max_torque)
        theta, theta_dot = self.state
        theta_dotdot = (u - self.m * self.g * self.l * mx.sin(theta)) / self.I
        theta_dot += self.dt * theta_dotdot
        theta_dot = mx.clip(theta_dot, -self.max_theta_dot, self.max_theta_dot)
        theta += self.dt * theta_dot
        theta = ((theta + mx.pi) % (2 * mx.pi)) - mx.pi

        self.state = mx.array([theta, theta_dot])
        observation = self.observe()
        reward = self.compute_reward(theta, theta_dot, u)
        done = abs(theta) > self.theta_threshold
        return (
            observation,
            reward,
            done,
            {
                "theta": theta,
                "theta_dot": theta_dot,
                "theta_dotdot": theta_dotdot,
                "action": u,
                "reward": reward,
            },
        )

    def compute_reward(self, theta, theta_dot, u):
        k_theta = 1.0
        k_theta_dot = 0.1
        k_action = 0.001

        angle_cost = k_theta * (theta**2)
        velocity_cost = k_theta_dot * (theta_dot**2)
        action_cost = k_action * (u**2)
        return float((1.0 - angle_cost - velocity_cost - action_cost))


if __name__ == "__main__":
    e = Env(m=2.0, l=5.0)
    e.reset()
    print(e.step(mx.array(1.2)))
