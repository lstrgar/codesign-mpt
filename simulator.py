import torch, numpy as np, taichi as ti

i32 = ti.i32
f32 = ti.f32
vec3 = ti.types.vector(3, f32)
vec2 = ti.types.vector(2, f32)

@ti.data_oriented
class Simulator:
    def __init__(self, config, workspace, nn, seed):

        self.config = config
        self.workspace_nms = len(workspace["masses"])
        self.workspace_nspr = len(workspace["springs"])
        self.nn = nn

        np.random.seed(seed)
        torch.random.manual_seed(seed)
        ti.init(
            arch=ti.cuda, 
            default_fp=f32,
            random_seed=seed,
            device_memory_fraction=0.95,
        )

        self.allocate_fields()
        self.initialize_workspace(workspace)

    #### Train / test ############################

    def train(self, batch):
        self.nn.train()
        self.prep_sim(batch)
        nn_inputs, nn_outputs = self.forward()
        self.compute_losses()
        self.average_losses()
        self.initialize_loss_grad()
        self.average_losses.grad()
        self.compute_losses.grad()
        self.backward(nn_inputs, nn_outputs)
        self.nn.learn()
        losses = self.losses.to_numpy()
        self.reset_sim()
        return losses

    def test(self, batch):
        self.nn.eval()
        self.prep_sim(batch)
        with torch.no_grad():
            _, _ = self.forward()
        self.compute_losses()
        losses = self.losses.to_numpy()
        self.reset_sim()
        return losses

    #### Forward / backward ############################

    def forward(self):
        torch_inputs = []
        torch_outputs = []
        self.compute_com(0, 0)
        for t in range(0, self.config["steps"]):
            self.compute_irradiance(t,)
            nn_input, nn_output = self.nn_forward(t)
            torch_inputs.append(nn_input)
            torch_outputs.append(nn_output)
            self.apply_spring_force(t)
            self.advance(t + 1)
        self.compute_com(self.config["steps"], 1)
        return torch_inputs, torch_outputs

    def backward(self, torch_inputs, torch_outputs):
        self.compute_com.grad(self.config["steps"], 1)
        for t in range(self.config["steps"]-1, -1, -1):
            self.advance.grad(t + 1)
            self.apply_spring_force.grad(t)
            self.nn_backward(t, torch_inputs[t], torch_outputs[t])
            self.compute_irradiance.grad(t)
        self.compute_com.grad(0, 0)

    #### Allocate taichi fields ############################

    def allocate_fields(self):
        # Terrain heightmap
        self.terrain_height = ti.field(dtype=f32, shape=(self.config["batch_size"], self.config["grid_dim"], self.config["grid_dim"]), needs_grad=self.config["needs_grad"])
        
        # Workspace
        self.masses = ti.Vector.field(3, dtype=f32, shape=self.workspace_nms, needs_grad=False)
        self.springs = ti.Vector.field(2, dtype=i32, shape=self.workspace_nspr, needs_grad=False)
        self.spring_l = ti.field(dtype=f32, shape=self.workspace_nspr, needs_grad=False)

        # Position and velocity
        self.x = ti.Vector.field(3, dtype=f32, shape=(self.config["batch_size"], self.config["steps"] + 1, self.workspace_nms), needs_grad=self.config["needs_grad"])
        self.init_z_shift = ti.field(dtype=f32, shape=(self.config["batch_size"]), needs_grad=False)
        self.v = ti.Vector.field(3, dtype=f32, shape=(self.config["batch_size"], self.config["steps"] + 1, self.workspace_nms), needs_grad=self.config["needs_grad"])
        self.vinc = ti.Vector.field(3, dtype=f32, shape=(self.config["batch_size"], self.config["steps"] + 1, self.workspace_nms), needs_grad=self.config["needs_grad"])
        
        # Irradiance
        self.irradiance = ti.field(dtype=f32, shape=(self.config["batch_size"], self.config["steps"], self.workspace_nms), needs_grad=self.config["needs_grad"])
        
        # Center of mass
        self.center_sum = ti.Vector.field(3, dtype=f32, shape=(self.config["batch_size"], 2), needs_grad=self.config["needs_grad"])
        self.center = ti.Vector.field(3, dtype=f32, shape=(self.config["batch_size"], 2), needs_grad=self.config["needs_grad"])
        
        # NN interface
        self.act = ti.field(dtype=f32, shape=(self.config["batch_size"], self.config["steps"], self.workspace_nspr), needs_grad=self.config["needs_grad"])
    
        # Actual masses and springs to simulate (subsets of workspace)
        self.batch_masses = ti.Vector.field(2, dtype=i32, shape=self.workspace_nms * self.config["batch_size"], needs_grad=False)
        self.nbm = ti.field(dtype=i32, shape=(), needs_grad=False)
        self.n_masses = ti.field(dtype=i32, shape=self.config["batch_size"], needs_grad=False)
        self.batch_springs = ti.Vector.field(2, dtype=i32, shape=self.workspace_nspr * self.config["batch_size"], needs_grad=False)
        self.nbs = ti.field(dtype=i32, shape=(), needs_grad=False)

        # Light source positions
        self.targets = ti.Vector.field(2, dtype=f32, shape=self.config["batch_size"], needs_grad=False)

        # Loss
        self.losses = ti.field(dtype=f32, shape=self.config["batch_size"], needs_grad=self.config["needs_grad"])
        self.total_loss = ti.field(dtype=f32, shape=(), needs_grad=self.config["needs_grad"])
        self.avg_loss = ti.field(dtype=f32, shape=(), needs_grad=self.config["needs_grad"])

    #### Initialize workspace ############################

    def initialize_workspace(self, workspace):
        masses = np.array(workspace["masses"], dtype=np.float32)
        center_point = self.config["grid_max"] / 2
        masses[:, 0] += center_point - masses[:, 0].mean()
        masses[:, 1] += center_point - masses[:, 1].mean()
        self.masses.from_numpy(np.ascontiguousarray(masses, dtype=np.float32))
        self.springs.from_numpy(np.ascontiguousarray(workspace["springs"], dtype=np.int32))
        self.spring_l.from_numpy(np.ascontiguousarray(workspace["spring_lengths"], dtype=np.float32))

    #### Load a batch ############################

    def prep_sim(self, batch):
        self.targets.from_numpy(batch["targets"])
        self.terrain_height.from_numpy(batch["terrain"])
        self.n_masses.from_numpy(batch["n_masses"])
        
        batch_masses = batch["masses"]
        self.nbm[None] = len(batch_masses)
        self.load_batch_masses(batch_masses)

        batch_springs = batch["springs"]
        self.nbs[None] = len(batch_springs)
        self.load_batch_springs(batch_springs)

        self.compute_body_shift()
        self.initialize_mass_pos()

    @ti.kernel
    def load_batch_masses(self, batch_masses: ti.types.ndarray()): # type: ignore
        for j in range(self.nbm[None]):
            self.batch_masses[j] = ti.Vector([batch_masses[j, 0], batch_masses[j, 1]], dt=i32)

    @ti.kernel
    def load_batch_springs(self, batch_springs: ti.types.ndarray()): # type: ignore
        for j in range(self.nbs[None]):
            self.batch_springs[j] = ti.Vector([batch_springs[j, 0], batch_springs[j, 1]], dt=i32)

    @ti.kernel
    def compute_body_shift(self): # type: ignore
        for j in range(self.nbm[None]):
            b, i = self.batch_masses[j]
            if self.masses[i][2] == 0.0:
                pt = ti.Vector([self.masses[i][0], self.masses[i][1]], dt=f32)
                ti.atomic_max(self.init_z_shift[b], self.ground_height_ti(b, pt))

    @ti.kernel
    def initialize_mass_pos(self): # type: ignore
        for j in range(self.nbm[None]):
            b, i = self.batch_masses[j]
            x, y, z = self.masses[i]
            self.x[b, 0, i] = ti.Vector([x, y, z + self.init_z_shift[b]])

    #### Simulator ############################

    @ti.kernel
    def compute_com(self, t: i32, ct: i32): # type: ignore
        for j in range(self.nbm[None]):
            b, i = self.batch_masses[j]
            ti.atomic_add(self.center_sum[b, ct], self.x[b, t, i])
        for b in range(self.config["batch_size"]):
            self.center[b, ct] = self.center_sum[b, ct] / self.n_masses[b]

    @ti.kernel
    def compute_irradiance(self, t: i32): # type: ignore
        for j in range(self.nbm[None]):
            b, i = self.batch_masses[j]
            target = ti.Vector([self.targets[b][0], self.targets[b][1], self.ground_height_ti(b, self.targets[b].xy)])
            dist = ti.math.distance(self.x[b, t, i], target)
            sensor_irradiance = (1 / (ti.sqrt(dist + self.config["eps"]) + self.config["eps"])) # Appendix A.5
            self.irradiance[b, t, i] = sensor_irradiance

    @ti.kernel
    def irrad_2_torch(self, t: i32, torch_input: ti.types.ndarray()): # type: ignore
        for b, i in ti.ndrange(self.config["batch_size"], self.workspace_nms):
            torch_input[b, i] = self.irradiance[b, t, i]

    def nn_forward(self, t):
        torch_input = torch.zeros(self.config["batch_size"], self.workspace_nms, 
                            device="cuda", dtype=torch.float32, requires_grad=self.config["needs_grad"])
        self.irrad_2_torch(t, torch_input)
        torch_output = self.nn(torch_input, t)
        if self.config["needs_grad"]:
            torch_output.grad = torch.zeros_like(torch_output, device="cuda", dtype=torch.float32)
        self.torch_act_2_taichi(t, torch_output)
        return torch_input, torch_output

    @ti.kernel
    def torch_act_2_taichi(self, t: i32, act_torch: ti.types.ndarray()): # type: ignore
        for j in range(self.nbs[None]):
            b, i = self.batch_springs[j]
            self.act[b, t, i] = act_torch[b, i]

    @ti.kernel
    def apply_spring_force(self, t: i32): # type: ignore
        for j in range(self.nbs[None]):
            bs, i = self.batch_springs[j]
            endpoint1 = self.springs[i][0]
            endpoint2 = self.springs[i][1]
            dist = self.x[bs, t, endpoint1] - self.x[bs, t, endpoint2]
            length = dist.norm()
            target_length = self.spring_l[i] * (1 + self.act[bs, t, i] * self.config["spring_a"])
            force = (length - target_length) * self.config["spring_k"] * dist / (length + self.config["eps"])
            impulse = self.config["dt"] * force
            ti.atomic_add(self.vinc[bs, t+1, endpoint1], -impulse)
            ti.atomic_add(self.vinc[bs, t+1, endpoint2], impulse)

    @ti.kernel
    def advance(self, t: i32): # type: ignore
        for j in range(self.nbm[None]):
            b, i = self.batch_masses[j]
            damping = ti.exp(-self.config["dt"] * self.config["drag_damping"])
            g = self.config["dt"] * -9.8 * ti.Vector([0.0, 0.0, 1.0])
            newv = damping * self.v[b, t-1, i] + g + self.vinc[b, t, i]
            oldx = self.x[b, t-1, i]
            newx = oldx + self.config["dt"] * newv
            if newx[2] < self.ground_height_ti(b, newx.xy):
                toi = self.estimate_toi(b, 0.0, self.config["dt"], oldx, newv)
                newx_toi = oldx + toi * newv
                newx_toi[2] = self.ground_height_ti(b, newx_toi.xy)
                normal = self.ground_normal(b, newx_toi.xy)
                newv_contact = self.v_on_contact(newv, normal)
                newx_contact = newx_toi + (self.config["dt"] - toi) * newv_contact
                newx_contact[2] = ti.math.max(newx_contact[2], self.ground_height_ti(b, newx_contact.xy))
                newx = newx_contact
                newv = newv_contact
            self.x[b, t, i] = newx
            self.v[b, t, i] = newv

    def nn_backward(self, t, torch_input, torch_output):
        act_grad_torch = torch.zeros(self.config["batch_size"], self.workspace_nspr, device="cuda", 
                            dtype=torch.float32, requires_grad=True)
        self.act_grad_2_torch(t, act_grad_torch)
        torch_output.backward(gradient=act_grad_torch)
        self.torch_input_grad_2_taichi(t, torch_input.grad)

    @ti.kernel
    def act_grad_2_torch(self, t: i32, grad_torch: ti.types.ndarray()): # type: ignore
        for j in range(self.nbs[None]):
            b, i = self.batch_springs[j]
            grad_torch[b, i] = self.act.grad[b, t, i]

    @ti.kernel
    def torch_input_grad_2_taichi(self, t: i32, grad_torch: ti.types.ndarray()): # type: ignore
        for b, i in ti.ndrange(self.config["batch_size"], self.workspace_nms):
            self.irradiance.grad[b, t, i] = grad_torch[b, i]

    #### Compute loss ############################

    @ti.kernel
    def initialize_loss_grad(self):
        self.avg_loss.grad[None] = 1.0

    @ti.kernel
    def compute_losses(self): # type: ignore
        for b in range(self.config["batch_size"]):
            dend = ti.math.distance(self.center[b, 1].xy, self.targets[b].xy)
            dbeg = ti.math.distance(self.center[b, 0].xy, self.targets[b].xy)
            l = dend / (dbeg + self.config["eps"])
            self.losses[b] = l
            ti.atomic_add(self.total_loss[None], l)

    @ti.kernel
    def average_losses(self):
        self.avg_loss[None] = self.total_loss[None] / self.config["batch_size"]

    #### Ground contact ############################

    @ti.func
    def ground_height_ti(self, b: i32, pt: vec2) -> f32: # type: ignore
        pt_x, pt_y = pt[0], pt[1]
        pt_x = ti.math.min(ti.math.max(0, pt_x), self.config["grid_max"])
        pt_y = ti.math.min(ti.math.max(0, pt_y), self.config["grid_max"])
        i = pt_x / (self.config["grid_max"] / (self.config["grid_dim"] - 1))
        j = pt_y / (self.config["grid_max"] / (self.config["grid_dim"] - 1))
        i1, j1 = ti.math.floor(i, dtype=i32), ti.math.floor(j, dtype=i32)
        frac_x, frac_y = i - i1, j - j1
        i2 = ti.math.min(i1 + 1, self.config["grid_dim"]-1)
        j2 = ti.math.min(j1 + 1, self.config["grid_dim"]-1)
        h00 = self.terrain_height[b, j1, i1]
        h10 = self.terrain_height[b, j1, i2]
        h01 = self.terrain_height[b, j2, i1]
        h11 = self.terrain_height[b, j2, i2]
        h = (h00 * (1 - frac_x) * (1 - frac_y) +
            h10 * frac_x * (1 - frac_y) +
            h01 * (1 - frac_x) * frac_y +
            h11 * frac_x * frac_y)
        return h

    @ti.func
    def ground_normal(self, b: i32, pt: vec2) -> vec3: # type: ignore
        delta = self.config["gn_delta"]
        h = self.ground_height_ti(b, pt)
        pt_x, pt_y = pt[0], pt[1]
        dxplus = self.ground_height_ti(b, ti.Vector([ti.min(pt_x+delta, self.config["grid_max"]), pt_y], dt=f32)) - h
        dxminus = h - self.ground_height_ti(b, ti.Vector([ti.max(pt_x-delta, 0.0), pt_y], dt=f32))
        dx = 0.5 * (dxplus + dxminus)
        dyplus = self.ground_height_ti(b, ti.Vector([pt_x, ti.min(pt_y+delta, self.config["grid_max"])], dt=f32)) - h
        dyminus = h - self.ground_height_ti(b, ti.Vector([pt_x, ti.max(pt_y-delta, 0.0)], dt=f32))
        dy = 0.5 * (dyplus + dyminus)
        return ti.math.cross(ti.Vector([delta, 0, dx], dt=f32), ti.Vector([0, delta, dy], dt=f32)).normalized()

    @ti.func
    def v_on_contact(self, v_old: ti.types.vector(3, ti.f32), normal: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32): 
        vn = v_old.dot(normal) * normal
        vn_mag = vn.norm()
        vt = v_old - vn
        vnew = ti.Vector([0.0, 0.0, 0.0], dt=f32)
        vt_mag = vt.norm()
        if vt_mag > 0.0:
            friction_mag = ti.math.clamp(self.config["friction"] * vn_mag, 0.0, vt_mag * 0.95)
            vf = -friction_mag * vt.normalized()
            vnew += vt + vf
        return vnew
    
    @ti.func
    def estimate_toi(self, b: i32, lower_bound: f32, upper_bound: f32, oldx: vec3, newv: vec3) -> f32: # type: ignore
        for _ in ti.static(range(self.config["toi_iter"])):
            mid_t = (lower_bound + upper_bound) * 0.5
            newx = oldx + mid_t * newv
            ground_height = self.ground_height_ti(b, newx.xy)
            if newx[2] < ground_height:
                upper_bound = mid_t
            else:
                lower_bound = mid_t
        return lower_bound
    
    #### Reset ############################

    @ti.kernel
    def reset_sim_state(self):
        self.x.fill(0.0)
        self.init_z_shift.fill(0.0)
        self.irradiance.fill(0.0)
        self.center_sum.fill(0.0)
        self.center.fill(0.0)
        self.v.fill(0.0)
        self.vinc.fill(0.0)
        self.act.fill(0.0)
        self.targets.fill(0.0)
        self.terrain_height.fill(0.0)
        self.batch_masses.fill(0)
        self.batch_springs.fill(0)
        self.nbm[None] = 0
        self.nbs[None] = 0
        self.n_masses.fill(0)

    @ti.kernel
    def zero_grads(self):
        self.x.grad.fill(0.0)
        self.irradiance.grad.fill(0.0)
        self.center_sum.grad.fill(0.0)
        self.center.grad.fill(0.0)
        self.v.grad.fill(0.0)
        self.vinc.grad.fill(0.0)
        self.act.grad.fill(0.0)
        self.terrain_height.grad.fill(0.0)
        self.losses.grad.fill(0.0)
        self.total_loss.grad[None] = 0.0
        self.avg_loss.grad[None] = 0.0

    @ti.kernel
    def zero_loss(self):
        self.losses.fill(0.0)
        self.total_loss[None] = 0.0
        self.avg_loss[None] = 0.0

    def reset_sim(self):
        self.reset_sim_state()
        if self.config["needs_grad"]:
            self.zero_grads()
        self.zero_loss()