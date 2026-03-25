import matplotlib.pyplot as plt

def plot_U(U, plot_indices, text, cmap='viridis'):
    for index in plot_indices:
        plt.contourf(U[index,:,:].cpu(), cmap = cmap, levels=255)
        plt.colorbar()
        plt.title(text +' ' + str(index))
        plt.show()
        
def plot_multidimensional_slices(Ut, sample_indices, time_steps):
    """
    Plots slices from a multidimensional array Ut using contourf.

    Parameters:
        Ut (numpy.ndarray): A multidimensional array of shape (n_sample, n_time_steps, nx, ny).
        sample_indices (list): A list of sample indices [i_0, i_1, ..., i_k].
        time_steps (list): A list of 15 time steps [t_0, t_1, ..., t_14].
    """
    if len(time_steps) != 15:
        raise ValueError("The list of time steps must have exactly 15 elements.")

    for sample_idx in sample_indices:
        if sample_idx >= Ut.shape[0]:
            raise IndexError(f"Sample index {sample_idx} is out of bounds for the array with {Ut.shape[0]} samples.")

        # Create a new figure for each sample index
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(f"Sample {sample_idx}", fontsize=16)

        # Loop over time steps and create subplots
        for idx, t in enumerate(time_steps):
            if t >= Ut.shape[1]:
                raise IndexError(f"Time step {t} is out of bounds for the array with {Ut.shape[1]} time steps.")

            ax = axes[idx // 5, idx % 5]
            
            # Extract the slice for the given sample and time step
            slice_data = Ut[sample_idx, t, :, :]

            # Plot using contourf
            contour = ax.contourf(slice_data.cpu(), cmap="viridis")

            # Add a title to each subplot
            ax.set_title(f"\u0394t = {t}", fontsize=10)

            # Remove axis ticks for better visualization
            ax.axis("off")

        # Adjust layout and add a colorbar
        fig.subplots_adjust(hspace=0.4, wspace=0.3)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(contour, cax=cbar_ax)

        # Show the plot for the current sample index
        plt.show()