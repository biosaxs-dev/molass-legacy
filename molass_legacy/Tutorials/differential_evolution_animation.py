"""
    Tutorials.differential_evolution_animation.py

    Differential Evolution Algorithm Animation
    Shows how the population evolves to find the global minimum
    
    Matches the Basin-Hopping animation style for direct comparison

    Copyright (c) 2026, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from numpy import exp, sqrt, cos, e, pi


# Ackley objective function (same as Basin-Hopping animation)
def ackley_objective(v):
    """
    Ackley function - matches bh-2-1.py for direct comparison
    Global minimum at (0, 0) with f = 0
    """
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2))) - exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20


class DifferentialEvolutionAnimator:
    """
    Visualize Differential Evolution algorithm on 2D test functions
    """
    
    def __init__(self, objective_func, bounds, NP=15, F=0.8, CR=0.9, max_iter=50):
        """
        Parameters:
        -----------
        objective_func : callable
            Function to minimize f(x, y) -> scalar
        bounds : tuple
            ((x_min, x_max), (y_min, y_max))
        NP : int
            Population size (number of agents)
        F : float
            Differential weight (mutation scale factor)
        CR : float
            Crossover probability
        max_iter : int
            Maximum number of generations
        """
        self.func = objective_func
        self.bounds = bounds
        self.NP = NP
        self.F = F
        self.CR = CR
        self.max_iter = max_iter
        
        # Initialize population randomly
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        self.population = np.random.uniform(
            [x_min, y_min], 
            [x_max, y_max], 
            (NP, 2)
        )
        
        # Evaluate initial fitness
        self.fitness = np.array([self.func(agent) for agent in self.population])
        
        # History for animation
        self.history = [self.population.copy()]
        self.fitness_history = [self.fitness.copy()]
        self.mutation_vectors = []  # Store mutation info for visualization
        
        # Best solution tracking
        self.best_idx = np.argmin(self.fitness)
        self.best_solution = self.population[self.best_idx].copy()
        self.best_fitness = self.fitness[self.best_idx]
        
    def evolve_one_generation(self):
        """
        Perform one generation of DE
        Returns mutation info for visualization
        """
        new_population = self.population.copy()
        mutations = []  # (agent_idx, trial_vector, accepted)
        
        for i in range(self.NP):
            # Select three random agents (distinct from i)
            candidates = [j for j in range(self.NP) if j != i]
            a_idx, b_idx, c_idx = np.random.choice(candidates, 3, replace=False)
            
            a = self.population[a_idx]
            b = self.population[b_idx]
            c = self.population[c_idx]
            
            # Mutation: v = a + F * (b - c)
            mutant = a + self.F * (b - c)
            
            # Crossover
            trial = self.population[i].copy()
            R = np.random.randint(0, 2)  # Ensure at least one dimension is replaced
            
            for d in range(2):
                if np.random.rand() < self.CR or d == R:
                    trial[d] = mutant[d]
            
            # Bound constraint
            trial = np.clip(trial, 
                          [self.bounds[0][0], self.bounds[1][0]], 
                          [self.bounds[0][1], self.bounds[1][1]])
            
            # Selection
            trial_fitness = self.func(trial)
            accepted = trial_fitness <= self.fitness[i]
            
            if accepted:
                new_population[i] = trial
                self.fitness[i] = trial_fitness
                
                # Update best
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial.copy()
                    self.best_idx = i
            
            mutations.append((i, trial, accepted, (a_idx, b_idx, c_idx)))
        
        self.population = new_population
        self.history.append(self.population.copy())
        self.fitness_history.append(self.fitness.copy())
        self.mutation_vectors.append(mutations)
        
    def run(self):
        """Run the full optimization"""
        for gen in range(self.max_iter):
            self.evolve_one_generation()
            if gen % 10 == 0:
                print(f"Generation {gen}: Best f = {self.best_fitness:.6f}")
        
        print(f"\nOptimization complete!")
        print(f"Best solution: x = {self.best_solution[0]:.6f}, y = {self.best_solution[1]:.6f}")
        print(f"Best fitness: f = {self.best_fitness:.6f}")
        
    def create_animation(self, show_mutations=True, show_3d=True, interval=300):
        """
        Create matplotlib animation matching BH animation style
        
        Parameters:
        -----------
        show_mutations : bool
            If True, show mutation vectors as arrows
        show_3d : bool
            If True, show 3D surface plot (like BH animation)
        interval : int
            Delay between frames in milliseconds
        """
        # Prepare grid for contour plot (same resolution as BH animation)
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        x = np.arange(x_min, x_max, 0.1)
        y = np.arange(y_min, y_max, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.func((X[i, j], Y[i, j]))
        
        # Create figure (match BH layout)
        fig = plt.figure(figsize=(16, 8))
        
        if show_3d:
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
            fig.suptitle("Demonstration of Differential Evolution Algorithm", fontsize=20)
            ax1.set_title("Surface Plot with Population", fontsize=16)
            ax2.set_title("Contour Plot with Population", fontsize=16)
            
            # 3D surface (same as BH animation)
            ax1.plot_surface(X, Y, Z, cmap='jet', alpha=0.3)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('f(x,y)')
        else:
            ax2 = fig.add_subplot(111)
            fig.suptitle("Demonstration of Differential Evolution Algorithm", fontsize=20)
            ax2.set_title("Contour Plot with Population", fontsize=16)
        
        fig.tight_layout()
        
        # 2D contour plot (same colormap as BH animation)
        ax2.contourf(X, Y, Z, cmap='jet', alpha=0.3)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.grid(True, alpha=0.3)
        
        # Initialize scatter plot (colored by fitness, matching BH colors)
        pop0 = self.history[0]
        fit0 = self.fitness_history[0]
        
        # 2D scatter plot
        scatter2d = ax2.scatter(pop0[:, 0], pop0[:, 1], 
                               c=fit0, s=100, cmap='hot_r', 
                               edgecolor='black', linewidth=1.5,
                               vmin=Z.min(), vmax=Z.min() + (Z.max() - Z.min()) * 0.5,
                               zorder=5)
        
        # 3D scatter plot (if enabled)
        if show_3d:
            scatter3d = ax1.scatter(pop0[:, 0], pop0[:, 1], fit0,
                                   c=fit0, s=100, cmap='hot_r',
                                   edgecolor='black', linewidth=1.5,
                                   vmin=Z.min(), vmax=Z.min() + (Z.max() - Z.min()) * 0.5,
                                   zorder=5)
        
        # Mark the global best (cyan marker like BH animation)
        best_marker2d, = ax2.plot([], [], 'o', color='cyan', markersize=15, 
                                 markeredgecolor='black', linewidth=2,
                                 label='Best', zorder=6)
        
        if show_3d:
            best_marker3d, = ax1.plot([], [], [], 'o', color='cyan', markersize=15,
                                     markeredgecolor='black', linewidth=2,
                                     label='Best', zorder=6)
        
        # Arrow patches for mutations (2D only for clarity)
        arrows = []
        
        # Generation text
        gen_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                           fontsize=12)
        
        ax2.legend(loc='upper right')
        
        def init():
            scatter2d.set_offsets(pop0)
            scatter2d.set_array(fit0)
            if show_3d:
                scatter3d._offsets3d = (pop0[:, 0], pop0[:, 1], fit0)
                scatter3d.set_array(fit0)
                best_marker3d.set_data([], [])
                best_marker3d.set_3d_properties([], 'z')
            best_marker2d.set_data([], [])
            gen_text.set_text('')
            return (scatter2d, best_marker2d, gen_text) if not show_3d else (scatter2d, scatter3d, best_marker2d, best_marker3d, gen_text)
        
        def update(frame):
            # Update population scatter
            pop = self.history[frame]
            fit = self.fitness_history[frame]
            scatter2d.set_offsets(pop)
            scatter2d.set_array(fit)
            
            if show_3d:
                scatter3d._offsets3d = (pop[:, 0], pop[:, 1], fit)
                scatter3d.set_array(fit)
            
            # Update best marker
            best_idx = np.argmin(fit)
            best_marker2d.set_data([pop[best_idx, 0]], [pop[best_idx, 1]])
            
            if show_3d:
                best_marker3d.set_data([pop[best_idx, 0]], [pop[best_idx, 1]])
                best_marker3d.set_3d_properties([fit[best_idx]], 'z')
            
            # Clear old arrows
            for arrow in arrows:
                arrow.remove()
            arrows.clear()
            
            # Draw mutation vectors if requested and available
            if show_mutations and frame > 0 and frame <= len(self.mutation_vectors):
                mutations = self.mutation_vectors[frame - 1]
                for i, trial, accepted, (a_idx, b_idx, c_idx) in mutations:
                    if accepted and i < 5:  # Show only first 5 to avoid clutter
                        # Draw arrow from old position to new (trial)
                        old_pos = self.history[frame - 1][i]
                        arrow = FancyArrowPatch(
                            old_pos, trial,
                            arrowstyle='->', 
                            color='lime',
                            alpha=0.6,
                            linewidth=2,
                            mutation_scale=20
                        )
                        ax2.add_patch(arrow)
                        arrows.append(arrow)
            
            # Update generation text
            best_fitness_curve = [np.min(f) for f in self.fitness_history]
            gen_text.set_text(f'Generation: {frame}\nBest f: {best_fitness_curve[frame]:.6f}\nPopulation: {self.NP} agents')
            
            return (scatter2d, best_marker2d, gen_text) if not show_3d else (scatter2d, scatter3d, best_marker2d, best_marker3d, gen_text)
        
        anim = FuncAnimation(fig, update, init_func=init,
                           frames=len(self.history), 
                           interval=interval, blit=False, repeat=True)
        
        # Save animation to GIF file
        print("Saving animation to differential_evolution_animation.gif...")
        anim.save("differential_evolution_animation.gif", writer='pillow', fps=1000/interval)
        print("Animation saved!")
        
        plt.show()
        return anim


# Additional test functions (optional)
def sphere_function(x, y):
    """Simple sphere function - convex, single minimum"""
    return x**2 + y**2


def rosenbrock_function(x, y):
    """Rosenbrock function - narrow valley"""
    a = 1
    b = 100
    return (a - x)**2 + b * (y - x**2)**2


if __name__ == "__main__":
    print("=" * 60)
    print("Differential Evolution Animation")
    print("Matches Basin-Hopping animation (bh-2-1.py) for comparison")
    print("=" * 60)
    
    # Use same function and bounds as BH animation
    func = ackley_objective
    bounds = ((-5.0, 5.0), (-5.0, 5.0))
    func_name = "Ackley"
    
    print(f"\nOptimizing {func_name} function...")
    print(f"Bounds: x ∈ [{bounds[0][0]}, {bounds[0][1]}], y ∈ [{bounds[1][0]}, {bounds[1][1]}]")
    
    # Create animator with DE parameters
    de = DifferentialEvolutionAnimator(
        objective_func=func,
        bounds=bounds,
        NP=15,      # Population size
        F=0.8,      # Differential weight
        CR=0.9,     # Crossover probability
        max_iter=50
    )
    
    # Run optimization
    print("\nRunning optimization...")
    de.run()
    
    # Create animation (match BH style with 3D surface)
    print("\nCreating animation...")
    print("(Close the plot window to finish)")
    anim = de.create_animation(show_mutations=True, show_3d=True, interval=300)
    
    print("\nAnimation complete!")
    print("\n" + "=" * 60)
    print("Comparison with Basin-Hopping:")
    print("- BH: Single point jumping between basins")
    print("- DE: Population of agents evolving cooperatively")
    print("- Same Ackley function and bounds")
    print("- Green arrows show accepted mutations (first 5 agents)")
    print("- Cyan marker = current best solution")
    print("=" * 60)
