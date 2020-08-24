import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
from matplotlib import animation
from itertools import combinations

# https://www.csuohio.edu/sites/default/files/3-%202015.pdf - 2 particles in 2 particles out reaction kinetics, still works for 2 particles in 1 out

class Particle:

    def __init__(self, x, y, vx, vy, radius = 0.01, styles = None, mat = 0):

        self.r = np.array((x, y))
        self.v = np.array((vx, vy))
        self.radius = radius
        self.reacted = False
        self.mat = mat
        self.styles = styles

        if self.mat == 0:
            # Default circle styles
            self.styles = {'edgecolor': 'tab:blue', 'fill': False}
        if self.mat == 1:
            self.styles = {'edgecolor': 'tab:green', 'fill': False}
        if self.mat == 2:
            self.styles = {'edgecolor': 'tab:red', 'fill': False}

    @property
    def x(self):
        return self.r[0]
    @x.setter
    def x(self, value):
        self.r[0] = value
    @property
    def y(self):
        return self.r[1]
    @y.setter
    def y(self, value):
        self.r[1] = value
    @property
    def vx(self):
        return self.v[0]
    @vx.setter
    def vx(self, value):
        self.v[0] = value
    @property
    def vy(self):
        return self.v[1]
    @vy.setter
    def vy(self, value):
        self.v[1] = value

    def overlaps(self, other):
        """Does the circle of this Particle overlap that of other?"""

        return np.hypot(*(self.r - other.r)) < self.radius + other.radius

    def draw(self, ax):
        """Add this Particle's Circle patch to the Matplotlib Axes ax."""

        circle = Circle(xy=self.r, radius=self.radius, **self.styles)
        ax.add_patch(circle)
        return circle

    def advance(self, dt):
        """Advance the Particle's position forward in time by dt."""

        self.bounce = False
        self.r += self.v * dt

        # Make the Particles bounce off the walls
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
            self.bounce = True
        if self.x + self.radius > 1:
            self.x = 1-self.radius
            self.vx = -self.vx
            self.bounce = True
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
            self.bounce = True
        if self.y + self.radius > 1:
            self.y = 1-self.radius
            self.vy = -self.vy
            self.bounce = True
class Simulation:
    """A class for a simple hard-circle molecular dynamics simulation.

    The simulation is carried out on a square domain: 0 <= x < 1, 0 <= y < 1.

    """

    def __init__(self, n, activate, radius=0.01, styles=None, temp = 0.2, show = True):
        """Initialize the simulation with n Particles with radii radius.

        radius can be a single value or a sequence with n values.

        Any key-value pairs passed in the styles dictionary will be passed
        as arguments to Matplotlib's Circle patch constructor when drawing
        the Particles.

        """
        self.show = show
        
        def on(event):
            self.react = True

        def off(event):
            self.react = False

        self.temp = temp
        self.n = n
        self.radius = radius
        self.react = False
        self.activate = activate

        if self.show:

            self.fig, [self.ax, self.ax3] = plt.subplots(ncols = 2)
            axon = plt.axes([0.7, 0.05, 0.1, 0.05])
            axoff = plt.axes([0.81, 0.05, 0.1, 0.05])
            axtxt = plt.axes([0.75, 0.9, 0.1, 0.05])
            for s in ['top','bottom','left','right']:
                axtxt.spines[s].set_linewidth(0)
            axtxt.xaxis.set_ticks([])
            axtxt.yaxis.set_ticks([])
            self.txt = plt.text(.96,.94,"Reacting = {}".format(self.react), bbox={'facecolor':'w','pad':5},
                     ha="right", va="top", transform=axtxt.transAxes)
            self.ax3.set_ylim([0, 50])
            self.ax3.set_xlim([0, 50])
            self.bon = Button(axon, "Start Reaction")
            self.boff = Button(axoff, "Stop Reaction")
            self.bon.on_clicked(on)
            self.boff.on_clicked(off)


    def init_particles(self, n, radius, styles=None):
        """Initialize the n Particles of the simulation.

        Positions and velocities are chosen randomly; radius can be a single
        value or a sequence with n values.

        """

        try:
            iterator = iter(radius)
            assert n == len(radius)
        except TypeError:
            # r isn't iterable: turn it into a generator that returns the
            # same value n times.
            def r_gen(n, radius):
                for i in range(n):
                    yield radius
            radius = r_gen(n, radius)
            
        self.n = n
        self.particles = []
        self.circles = []
        for i, rad in enumerate(radius):
            if i < n/2:
                mat = 0
            else:
                mat = 1
            # Try to find a random initial position for this particle.
            while True:
                # Choose x, y so that the Particle is entirely inside the
                # domain of the simulation.
                x, y = rad + (1 - 2*rad) * np.random.random_sample(2)
                # Choose a random velocity (within some reasonable range of
                # values) for the Particle.
                vr = 0.05 * np.random.random_sample() + self.temp
                vphi = 2*np.pi * np.random.random_sample()
                vx, vy = vr * np.cos(vphi), vr * np.sin(vphi)
                particle = Particle(x, y, vx, vy, rad, styles, mat)
                # Check that the Particle doesn't overlap one that's already
                # been placed.
                for p2 in self.particles:
                    if p2.overlaps(particle):
                        break
                else:
                    self.particles.append(particle)
                    break
    
    def handle_collisions(self):
        """

        Detect and handle any collisions between the Particles.

        When two Particles collide, they do so elastically: their velocities
        change such that both energy and momentum are conserved.

        """

        def change_velocities(p1, p2):
            """

            Particles p1 and p2 have collided elastically: update their
            velocities.

            """

            m1, m2 = p1.radius**2, p2.radius**2
            M = m1 + m2
            r1, r2 = p1.r, p2.r
            d = np.linalg.norm(r1 - r2)**2
            v1, v2 = p1.v, p2.v
            u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
            u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)
            p1.v = u1
            p2.v = u2
            
            s = p1.r-p2.r
            p1.r = p2.r + (p1.radius + p2.radius) * s / np.linalg.norm(s)

        def react(pairs):
            newl = []
            dels = []
            total = []
            for i in pairs:
                rem = False
                for j in i:
                    if j in total:
                        pairs.remove(i)
                        rem = True
                if not rem:
                    total += i      
            for i in pairs:
                p1, p2 = i
                new = Particle(np.average((p1.x, p2.x)),
                               np.average((p1.y, p2.y)),
                               np.average((p1.vx, p2.vx)),
                               np.average((p1.vy, p2.vy)),
                               radius = np.sqrt(p1.radius**2 + p2.radius**2),
                               styles = {'edgecolor': 'tab:red', 'linewidth': 1, 'fill': None},
                               mat = 2)
                new.reacted = True
                newl.append(new)
                dels += i

            return newl, dels
                

        """

        We're going to need a sequence of all of the pairs of particles when
        we are detecting collisions. combinations generates pairs of indexes
        into the self.particles list of Particles on the fly.

        """
        pairs = combinations(range(len(self.particles)), 2)
        reacts = []
        for i,j in pairs:
            if self.particles[i].overlaps(self.particles[j]):
                change_velocities(self.particles[i], self.particles[j])
                if self.react and not (self.particles[i].reacted or self.particles[j].reacted) and (self.particles[i].mat + self.particles[j].mat == 1):
                    energy_sum = 0.5*(np.linalg.norm(self.particles[i].v-self.particles[j].v)*50)**2
                    if energy_sum >= self.activate:
                        reacts.append([self.particles[i], self.particles[j]])
        a, b = react(reacts)
        for i in b:
            if self.show:
                del self.circles[self.particles.index(i)]
            self.particles.remove(i)
        for i in a:
            self.particles.append(i)
            if self.show:
                self.circles.append(i.draw(self.ax))

    def handle_splits(self):

        def split(p, activate):

            newp1 = Particle(0, 0, 0, 0, mat = 0)
            newp2 = Particle(0, 0, 0, 0, mat = 1)
            perpdir = np.array((1, -p.vx / p.vy))
            perpdir = perpdir / np.linalg.norm(perpdir)
            newp1.r = p.r - 0.01 * perpdir
            newp2.r = p.r + 0.01 * perpdir
            minhoriz = np.sqrt(activate*2)/100
            ranadd = np.abs(np.random.normal(scale = minhoriz/10))
            newp1.v = p.v - (minhoriz + ranadd) * perpdir
            newp2.v = p.v + (minhoriz + ranadd) * perpdir
            return newp1, newp2

        splitl = []
        delet = []

        if self.react:
            for p in self.particles:
                if p.mat == 2:
                    if np.random.random_sample() < 0.001:
                        delet.append(p)
                        for i in split(p, self.activate):
                            splitl.append(i)

        for i in delet:
            if self.show:
                del self.circles[self.particles.index(i)]
            self.particles.remove(i)
        for i in splitl:
            self.particles.append(i)
            if self.show:
                self.circles.append(i.draw(self.ax))
            


    def advance_animation(self, dt):
        """

        Advance the animation by dt, returning the updated Circles list.

        """

        if self.show:
            for i, p in enumerate(self.particles):
                if len(self.spds) != len(self.particles):
                    self.spds = np.array(list(i for i in range(len(self.particles))))
                self.spds[i] = np.linalg.norm(p.v)*100*(100*p.radius)**2

            x = [5, 15, 25, 35, 45]
            h = [0, 0, 0, 0, 0]

            self.hist = self.spds / 10 
            self.hist = np.floor(self.hist)
            for i in range(5):
                h[i] = sum(self.hist == i)

            for i, r in enumerate(self.bar.patches):
                r.set_height(h[i])

            self.txt.set_text("Reacting = {}".format(self.react))

        for i, p in enumerate(self.particles):
            p.advance(dt)
            if self.show:
                self.circles[i].center = p.r
        self.handle_collisions()
        self.handle_splits()

        if self.counter == 500:
            self.react = True

        if self.counter in list(self.out_times):
            for i in self.particles:
                self.graph[i.mat][int((self.counter - 500)*(self.freq-1)/1500)] += 1

        if self.counter == 2000:
            self.react = False

        self.counter += 1
        if self.show:
            return self.circles + self.bar.patches + [self.txt]

    def advance(self, dt):
        """

        Advance the animation by dt.

        """

        for i, p in enumerate(self.particles):
            p.advance(dt)
        self.handle_collisions()

    def init(self):
        """

        Initialize the Matplotlib animation.

        """

        self.circles = []
        for particle in self.particles:
            self.circles.append(particle.draw(self.ax)) 
        return self.circles + self.bar.patches + [self.txt]   

    def animate(self, i):
        """

        The function passed to Matplotlib's FuncAnimation routine.

        """

        self.advance_animation(0.02)
        return self.circles + self.bar.patches + [self.txt]

    def do_animation(self, save=False):
        """

        Set up and carry out the animation of the molecular dynamics.

        To save the animation as a MP4 movie, set save=True.

        """

        self.freq = 21
        self.init_particles(self.n, self.radius)
        self.counter = 0
        self.out_times = np.linspace(start = 500, stop = 2000, num = self.freq)
        for i in range(self.freq):
            self.out_times[i] = int(self.out_times[i])
        self.graph = np.zeros((3, self.freq))

        if self.show:
            for s in ['top','bottom','left','right']:
                self.ax.spines[s].set_linewidth(2)
            self.ax.set_aspect('equal', 'box')
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
            self.ax.xaxis.set_ticks([0, 1])
            self.ax.yaxis.set_ticks([0, 1])
            self.ax.set_ylabel("m")
            self.ax.set_xlabel("m")
            self.spds = np.zeros(self.n)
            for i, p in enumerate(self.particles):
                self.spds[i] = np.linalg.norm(p.v)*100

            x = [5, 15, 25, 35, 45]
            h = [0, 0, 0, 0, 0]

            self.hist = self.spds / 10 
            self.hist = np.floor(self.hist)
            for i in range(5):
                h[i] = sum(self.hist == i)

            self.bar = self.ax3.bar(x, h, width = 8)
            anim = animation.FuncAnimation(self.fig, self.animate, init_func=self.init,
                                frames=800, interval=2, blit=True)
            if save:
                writer = animation.FFMpegWriter(fps=100, bitrate=1800)
                anim.save('collision.mp4', writer=writer)
            else:
                plt.show()

        if not self.show:
            while self.counter <= 2002:
                self.advance_animation(0.02)
        


if __name__ == '__main__':
    nparticles = 50
    radius = 0.01
    styles = {'edgecolor': 'C0', 'linewidth': 1, 'fill': None}
    # sim1 = Simulation(nparticles, 100, radius, styles, 0.2, True)
    # sim1.do_animation()
    sims = list(Simulation(nparticles, 75, radius, styles, 0.1+0.01*i, show = False) for i in range(11))
    results = {}
    str_out = ""
    for i in sims:        
        for j in range(50):
            i.do_animation(save=False)
            if i.temp in results.keys():
                results[i.temp] += i.graph[2]
            else:
                results[i.temp] = i.graph[2]
        results[i.temp] = results[i.temp] / 1250
        str_out += "Temperature: {:.2f}\nPercentage which reacted: {}%\n".format(i.temp, results[i.temp]*100)

    print(str_out)

