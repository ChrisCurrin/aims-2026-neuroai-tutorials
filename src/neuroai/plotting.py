"""
Plotting helper functions for the NeuroAI tutorials.

This module contains functions to visualize:
- Input signals and spatial profiles
- Weight matrices (heatmaps)
- Simulation results (firing rates vs input)
- Spike raster plots
- Comparison between SNN and CNN outputs
- Hopfield network patterns and recall
- Interactive Hopfield network topology
- STDP Learning Windows
"""

import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots


def plot_input_profile(input_signal: torch.Tensor, num_neurons: int) -> None:
    """
    Plots the spatial profile of the input signal at the first time step.

    Args:
        input_signal (torch.Tensor): Input tensor of shape (num_steps, num_neurons).
        num_neurons (int): Number of neurons in the layer.
    """
    fig = px.line(
        x=np.arange(num_neurons),
        y=input_signal[0, :].numpy(),
        labels={"x": "Neuron Index", "y": "Input Intensity"},
        title="Input Spatial Profile (Edge)",
        markers=True,
    )
    fig.show()


def plot_weight_matrix(
    weights: torch.Tensor, title: str = "Lateral Inhibition Weight Matrix"
) -> None:
    """
    Visualizes a weight matrix as a heatmap.

    Args:
        weights (torch.Tensor): Weight matrix of shape (num_neurons, num_neurons).
        title (str, optional): Title of the plot. Defaults to "Lateral Inhibition Weight Matrix".
    """
    fig = px.imshow(
        weights.numpy(),
        color_continuous_scale="RdBu_r",
        labels=dict(x="Source Neuron (j)", y="Target Neuron (i)", color="Weight"),
        title=title,
    )
    fig.show()


def plot_simulation_results(
    input_signal: torch.Tensor, firing_rate: torch.Tensor, num_neurons: int
) -> None:
    """
    Plots the input intensity and output firing rate on a dual-axis plot to demonstrate edge enhancement.

    Args:
        input_signal (torch.Tensor): Input tensor of shape (num_steps, num_neurons).
        firing_rate (torch.Tensor): Average firing rate tensor of shape (num_neurons,).
        num_neurons (int): Number of neurons.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot Input Intensity
    fig.add_trace(
        go.Scatter(
            x=np.arange(num_neurons),
            y=input_signal[0].numpy(),
            name="Input Intensity",
            line=dict(color="blue", dash="dash"),
        ),
        secondary_y=False,
    )

    # Plot Firing Rate
    fig.add_trace(
        go.Scatter(
            x=np.arange(num_neurons),
            y=firing_rate.detach().numpy(),
            name="Output Firing Rate",
            mode="lines+markers",
            line=dict(color="red"),
        ),
        secondary_y=True,
    )

    # Add annotation for Edge Enhancement
    edge_idx = num_neurons // 2
    fig.add_annotation(
        x=edge_idx - 1,
        y=firing_rate[edge_idx - 1],
        text="Edge Enhancement",
        showarrow=True,
        arrowhead=1,
        yref="y2",
    )

    fig.update_layout(
        title_text="Edge Detection via Lateral Inhibition: The Mach Band Effect",
        xaxis_title="Neuron Index",
    )

    fig.update_yaxes(title_text="Input Intensity", secondary_y=False, title_font_color="blue")
    fig.update_yaxes(
        title_text="Firing Rate (spikes/step)", secondary_y=True, title_font_color="red"
    )

    fig.show()


def plot_raster(spk_rec: torch.Tensor) -> None:
    """
    Generates a raster plot of spike times.

    Args:
        spk_rec (torch.Tensor): Recorded spikes tensor of shape (num_steps, num_neurons).
    """
    spike_indices = torch.nonzero(spk_rec)
    time_steps = spike_indices[:, 0].numpy()
    neuron_indices = spike_indices[:, 1].numpy()

    fig_raster = px.scatter(
        x=time_steps,
        y=neuron_indices,
        labels={"x": "Time Step", "y": "Neuron Index"},
        title="Spike Raster Plot",
        template="simple_white",
    )
    fig_raster.update_traces(marker=dict(size=5, color="black", symbol="line-ns-open"))
    fig_raster.show()


def plot_cnn_comparison(
    firing_rate: torch.Tensor, conv_output: torch.Tensor, num_neurons: int
) -> None:
    """
    Compares the SNN firing rate with the output of a standard CNN convolution.

    Args:
        firing_rate (torch.Tensor): SNN firing rate tensor of shape (num_neurons,).
        conv_output (torch.Tensor): CNN output tensor of shape (num_neurons,).
        num_neurons (int): Number of neurons.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Plot SNN Firing Rate
    fig.add_trace(
        go.Scatter(
            x=np.arange(num_neurons),
            y=firing_rate.detach().numpy(),
            name="SNN Output",
            mode="lines+markers",
            line=dict(color="red"),
        ),
        secondary_y=False,
    )

    # Plot CNN Output
    fig.add_trace(
        go.Scatter(
            x=np.arange(num_neurons),
            y=conv_output.numpy(),
            name="CNN Output",
            line=dict(color="green", dash="dash"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title_text="Biological Mechanism vs. Deep Learning Operation",
        xaxis_title="Neuron Index",
    )

    fig.update_yaxes(title_text="SNN Firing Rate", secondary_y=False, title_font_color="red")
    fig.update_yaxes(
        title_text="CNN Convolution Output",
        secondary_y=True,
        title_font_color="green",
    )

    fig.show()


def plot_hopfield_patterns(p1: torch.Tensor, p2: torch.Tensor, n_sqrt: int) -> None:
    """
    Visualizes two stored patterns in the Hopfield network.

    Args:
        p1 (torch.Tensor): Flattened tensor of pattern 1.
        p2 (torch.Tensor): Flattened tensor of pattern 2.
        n_sqrt (int): Square root of the number of neurons (dimension of the image).
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Stored Pattern 1", "Stored Pattern 2"))

    fig.add_trace(
        px.imshow(p1.view(n_sqrt, n_sqrt).numpy(), color_continuous_scale="gray").data[0],
        row=1,
        col=1,
    )
    fig.add_trace(
        px.imshow(p2.view(n_sqrt, n_sqrt).numpy(), color_continuous_scale="gray").data[0],
        row=1,
        col=2,
    )

    fig.update_layout(title_text="Stored Patterns in Hopfield Network", height=400)
    fig.show()


def plot_hopfield_recall(
    p1: torch.Tensor,
    corrupted_p1: torch.Tensor,
    recovered_p1: torch.Tensor,
    n_sqrt: int,
    noise_level: float,
) -> None:
    """
    Visualizes the original, corrupted, and recovered patterns.

    Args:
        p1 (torch.Tensor): Original pattern tensor.
        corrupted_p1 (torch.Tensor): Corrupted pattern tensor.
        recovered_p1 (torch.Tensor): Recovered pattern tensor.
        n_sqrt (int): Image dimension.
        noise_level (float): The noise level used for corruption (for title).
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Original",
            f"Corrupted ({int(noise_level * 100)}% noise)",
            "Recovered",
        ),
    )

    fig.add_trace(
        px.imshow(p1.view(n_sqrt, n_sqrt).numpy(), color_continuous_scale="gray").data[0],
        row=1,
        col=1,
    )
    fig.add_trace(
        px.imshow(corrupted_p1.view(n_sqrt, n_sqrt).numpy(), color_continuous_scale="gray").data[0],
        row=1,
        col=2,
    )
    fig.add_trace(
        px.imshow(recovered_p1.view(n_sqrt, n_sqrt).numpy(), color_continuous_scale="gray").data[0],
        row=1,
        col=3,
    )

    fig.update_layout(title_text="Associative Memory Recall", height=400)
    fig.show()


def get_graph_data(
    num_nodes: int, hopfield_network_class: type
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[str],
    list[str],
]:
    """
    Generates graph layout data (nodes and edges) for a Hopfield network of a given size.
    Used for the interactive topology visualization.

    Args:
        num_nodes (int): Number of neurons in the network.
        hopfield_network_class (type): The class definition of the HopfieldNetwork to instantiate.

    Returns:
        tuple containing lists for:
        - Excitatory edge X coordinates
        - Excitatory edge Y coordinates
        - Inhibitory edge X coordinates
        - Inhibitory edge Y coordinates
        - Node X coordinates
        - Node Y coordinates
        - Node hover text
        - Node colors
    """
    # 1. Create and Train Network
    viz_net = hopfield_network_class(num_nodes)

    # // # Store a simple alternating pattern: [1, -1, 1, -1, ...]
    # // pattern = torch.ones(1, num_nodes)
    # // pattern[0, 1::2] = -1

    # Store a random pattern for more complex topology
    # Use a generator seeded with num_nodes to ensure consistency between frames and initial plot
    g = torch.Generator()
    g.manual_seed(num_nodes)
    pattern = torch.where(torch.randn(1, num_nodes, generator=g) > 0, 1.0, -1.0)
    viz_net.train(pattern)

    # 2. Generate Graph Layout
    G = nx.complete_graph(num_nodes)
    pos = nx.circular_layout(G)

    # 3. Extract Edges
    edge_x_exc, edge_y_exc = [], []
    edge_x_inh, edge_y_inh = [], []

    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = viz_net.weights[u, v].item()

        # Add edge segments (separated by None)
        if weight > 0:  # Excitatory (Red)
            edge_x_exc.extend([x0, x1, None])
            edge_y_exc.extend([y0, y1, None])
        elif weight < 0:  # Inhibitory (Blue)
            edge_x_inh.extend([x0, x1, None])
            edge_y_inh.extend([y0, y1, None])

    # 4. Extract Nodes
    node_x, node_y = [], []
    node_text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        state = pattern[0, node].item()
        node_text.append(f"Neuron {node}<br>State: {int(state)}")
        node_color.append("white" if state > 0 else "black")

    return (
        edge_x_exc,
        edge_y_exc,
        edge_x_inh,
        edge_y_inh,
        node_x,
        node_y,
        node_text,
        node_color,
    )


def plot_hopfield_topology(hopfield_network_class: type) -> None:
    """
    Creates an interactive Plotly figure with a slider to visualize the topology
    of a Hopfield network as the number of neurons increases.

    Args:
        hopfield_network_class (type): The class definition of the HopfieldNetwork.
    """
    # --- Pre-compute Frames for Slider ---
    frames = []
    steps = []
    node_counts = range(3, 13)  # Slider from 3 to 12 neurons

    for n in node_counts:
        ex_x, ex_y, in_x, in_y, n_x, n_y, n_txt, n_col = get_graph_data(n, hopfield_network_class)

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(x=ex_x, y=ex_y),  # Update Trace 0
                    go.Scatter(x=in_x, y=in_y),  # Update Trace 1
                    go.Scatter(
                        x=n_x, y=n_y, text=n_txt, marker=dict(color=n_col)
                    ),  # Update Trace 2
                ],
                name=str(n),
            )
        )

        steps.append(
            dict(
                method="animate",
                args=[
                    [str(n)],
                    dict(
                        mode="immediate",
                        frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0),
                    ),
                ],
                label=str(n),
            )
        )

    # --- Initial Plot (N=5) ---
    start_n = 5
    ex_x, ex_y, in_x, in_y, n_x, n_y, n_txt, n_col = get_graph_data(start_n, hopfield_network_class)

    fig = go.Figure(
        data=[
            # Trace 0: Excitatory Edges
            go.Scatter(
                x=ex_x,
                y=ex_y,
                mode="lines",
                line=dict(width=1, color="red"),
                name="Excitatory (>0)",
                hoverinfo="none",
            ),
            # Trace 1: Inhibitory Edges
            go.Scatter(
                x=in_x,
                y=in_y,
                mode="lines",
                line=dict(width=1, color="blue", dash="dot"),
                name="Inhibitory (<0)",
                hoverinfo="none",
            ),
            # Trace 2: Nodes
            go.Scatter(
                x=n_x,
                y=n_y,
                mode="markers+text",
                marker=dict(size=30, color=n_col, line=dict(width=2, color="black")),
                text=[str(i) for i in range(len(n_x))],
                textposition="middle center",
                hoverinfo="text",
                hovertext=n_txt,
                name="Neurons",
            ),
        ],
        layout=go.Layout(
            title="Hopfield Network Topology (Interactive)",
            width=600,
            height=600,
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            sliders=[
                dict(
                    active=node_counts.index(start_n),
                    currentvalue={"prefix": "Num Neurons: "},
                    pad={"t": 50},
                    steps=steps,
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.1,
                    y=1.15,
                    buttons=[
                        dict(
                            label="Play Animation",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=500, redraw=True),
                                    fromcurrent=True,
                                ),
                            ],
                        )
                    ],
                )
            ],
        ),
        frames=frames,
    )

    fig.show()


def plot_raster_from_list(trial_data: list[np.ndarray], title: str) -> None:
    """
    Generates a raster plot of spike times for multiple neurons.

    Args:
        trial_data (list[np.ndarray]): List of spike times for each neuron.
        title (str): Title of the plot.
    """
    fig = go.Figure()

    for n, spike_times in enumerate(trial_data):
        fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=[n] * len(spike_times),
                mode="markers",
                marker=dict(symbol="circle", size=4, color="black"),
                name=f"Neuron {n}",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Neuron Index",
        showlegend=False,
        height=600,
    )
    fig.show()


def plot_stdp_window(delta_t: np.ndarray, dw: np.ndarray) -> None:
    """
    Plots the STDP learning window.

    Args:
        delta_t (np.ndarray): Time differences (t_post - t_pre).
        dw (np.ndarray): Weight changes.
    """
    fig = go.Figure()

    # Main STDP curve
    fig.add_trace(
        go.Scatter(
            x=delta_t, y=dw, mode="lines", name="STDP Window", line=dict(color="black", width=2)
        )
    )

    # Fill areas
    fig.add_trace(
        go.Scatter(
            x=delta_t[delta_t > 0],
            y=dw[delta_t > 0],
            fill="tozeroy",
            mode="none",
            name="LTP (Pre → Post)",
            fillcolor="rgba(0, 255, 0, 0.3)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=delta_t[delta_t < 0],
            y=dw[delta_t < 0],
            fill="tozeroy",
            mode="none",
            name="LTD (Post → Pre)",
            fillcolor="rgba(255, 0, 0, 0.3)",
        )
    )

    # Layout
    fig.update_layout(
        title="STDP Learning Window",
        xaxis_title="Δt = t_post - t_pre (ms)",
        yaxis_title="Δw",
        showlegend=True,
        height=500,
    )

    # Add zero lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.show()


def plot_activation_functions(
    x: np.ndarray, y_sigmoid: np.ndarray, y_tanh: np.ndarray, y_relu: np.ndarray
) -> None:
    """
    Plots common activation functions: Sigmoid, Tanh, and ReLU.

    Args:
        x (np.ndarray): Input range array.
        y_sigmoid (np.ndarray): Sigmoid output array.
        y_tanh (np.ndarray): Tanh output array.
        y_relu (np.ndarray): ReLU output array.
    """
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Sigmoid", "Tanh", "ReLU"))

    fig.add_trace(
        go.Scatter(x=x, y=y_sigmoid, name="Sigmoid", line=dict(color="blue", width=2)), row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=x, y=y_tanh, name="Tanh", line=dict(color="red", width=2)), row=1, col=2
    )

    fig.add_trace(
        go.Scatter(x=x, y=y_relu, name="ReLU", line=dict(color="green", width=2)), row=1, col=3
    )

    fig.update_layout(title_text="Activation Functions", height=400, showlegend=False)

    fig.update_xaxes(title_text="Input")
    fig.update_yaxes(
        title_text="Output",
        range=[-1, x.max()],
    )

    fig.show()


def plot_neuron_population_response(population_outputs: np.ndarray) -> None:
    """
    Visualizes the response of a neuron population with a bar chart and histogram.

    Args:
        population_outputs (np.ndarray): Array of output values from the neuron population.
    """
    # Visualize the population response using Plotly
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Population Response", "Distribution of Outputs")
    )

    # Bar chart of neuron outputs
    fig.add_trace(
        go.Bar(x=list(range(len(population_outputs))), y=population_outputs, name="Neuron Output"),
        row=1,
        col=1,
    )

    # Add mean line to bar chart
    fig.add_hline(
        y=population_outputs.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text="Mean",
        row=1,
        col=1,
    )

    # Histogram of outputs
    fig.add_trace(
        go.Histogram(x=population_outputs, nbinsx=10, name="Distribution", marker_color="purple"),
        row=1,
        col=2,
    )

    fig.update_layout(height=500, showlegend=False, title_text="Neuron Population Analysis")

    fig.update_xaxes(title_text="Neuron Index", row=1, col=1)
    fig.update_yaxes(title_text="Output", row=1, col=1)

    fig.update_xaxes(title_text="Output Value", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)

    fig.show()


def plot_population_coding(intensities: np.ndarray, responses: np.ndarray) -> None:
    """
    Plots the population coding curve (mean response vs input intensity).

    Args:
        intensities (np.ndarray): Array of input intensity values.
        responses (np.ndarray): Array of mean population responses.
    """
    fig = px.line(
        x=intensities,
        y=responses,
        markers=True,
        labels={"x": "Input Intensity", "y": "Mean Population Response"},
        title="Population Coding: How Neural Activity Encodes Stimulus Intensity",
    )

    fig.update_layout(height=500)

    fig.show()


def plot_static_vs_dynamic(
    time: np.ndarray,
    input_signal: np.ndarray,
    perceptron_output: np.ndarray,
    membrane_potential: np.ndarray,
    spike_times: np.ndarray,
    threshold: float,
) -> None:
    """
    Plots a comparison between a static Perceptron and a dynamic Biological Neuron (LIF).

    Args:
        time (np.ndarray): Time array.
        input_signal (np.ndarray): Input signal array.
        perceptron_output (np.ndarray): Perceptron output array.
        membrane_potential (np.ndarray): Membrane potential array for biological neuron.
        spike_times (np.ndarray): Array of spike times.
        threshold (float): Firing threshold.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Input Signal",
            "Perceptron Output (Static)",
            "Biological Neuron (Dynamic Spiking)",
        ),
    )

    fig.add_trace(go.Scatter(x=time, y=input_signal, name="Input", fill="tozeroy"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=time, y=perceptron_output, name="Perceptron", line=dict(color="green")),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time, y=membrane_potential, name="Membrane Potential", line=dict(color="red")),
        row=3,
        col=1,
    )

    # Add spikes
    fig.add_trace(
        go.Scatter(
            x=spike_times,
            y=np.ones_like(spike_times) * threshold,
            mode="markers",
            marker=dict(symbol="line-ns-open", size=10, color="black"),
            name="Spikes",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(height=700, title_text="Artificial vs. Biological Neuron Dynamics")
    fig.show()


def plot_lif_simulation(
    time: np.ndarray,
    input_current: np.ndarray,
    v: np.ndarray,
    spike_times: np.ndarray,
    threshold: float,
) -> None:
    """
    Plots the results of a LIF neuron simulation.

    Args:
        time (np.ndarray): Time array.
        input_current (np.ndarray): Input current array.
        v (np.ndarray): Membrane potential array.
        spike_times (np.ndarray): Array of spike times.
        threshold (float): Firing threshold.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Input Current", "Membrane Potential & Spikes"),
        vertical_spacing=0.1,
    )

    # Input current
    fig.add_trace(
        go.Scatter(x=time, y=input_current, name="Current (I)", fill="tozeroy"), row=1, col=1
    )

    # Membrane potential
    fig.add_trace(
        go.Scatter(x=time, y=v, name="Membrane Potential (V)", line=dict(color="red")), row=2, col=1
    )

    # Spikes
    fig.add_trace(
        go.Scatter(
            x=spike_times,
            y=np.ones_like(spike_times) * -40,  # Arbitrary y-position for spikes above threshold
            mode="markers",
            marker=dict(symbol="line-ns-open", size=15, color="black"),
            name="Spikes",
        ),
        row=2,
        col=1,
    )

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="gray",
        annotation_text="Threshold",
        row=2,
        col=1,
    )

    fig.update_layout(height=600, title_text="LIF Neuron Simulation")
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
    fig.update_yaxes(title_text="Current (nA)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (mV)", row=2, col=1)

    fig.show()


def plot_synaptic_conductance(
    time: np.ndarray, g_syn: np.ndarray, spike_times: list | np.ndarray
) -> None:
    """
    Plots synaptic conductance and input spikes.

    Args:
        time (np.ndarray): Time array.
        g_syn (np.ndarray): Synaptic conductance array.
        spike_times (list | np.ndarray): List or array of spike times.
    """
    fig = go.Figure()

    # Plot conductance
    fig.add_trace(go.Scatter(x=time, y=g_syn, name="Conductance (g)", fill="tozeroy"))

    # Add markers for spike times
    spike_times_arr = np.array(spike_times)  # Ensure numpy array for plotting
    fig.add_trace(
        go.Scatter(
            x=spike_times_arr,
            y=np.zeros_like(spike_times_arr),
            mode="markers",
            marker=dict(symbol="triangle-up", size=15, color="red"),
            name="Input Spikes",
        )
    )

    fig.update_layout(
        title="Synaptic Conductance (Summation of EPSPs)",
        xaxis_title="Time (ms)",
        yaxis_title="Conductance (nS)",
        height=500,
    )

    fig.show()


# =============================================================================
# Introduction Tutorial Plotting Functions
# =============================================================================


def plot_activation_functions_comparison(
    z: np.ndarray,
    sigmoid_vals: np.ndarray,
    relu_vals: np.ndarray,
    tanh_vals: np.ndarray,
    same_range: bool = True,
) -> None:
    """
    Plots three common activation functions side-by-side.

    Args:
        z: Input values array.
        sigmoid_vals: Sigmoid output values.
        relu_vals: ReLU output values.
        tanh_vals: Tanh output values.
    """
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Sigmoid σ(z)", "ReLU", "Tanh"))

    fig.add_trace(
        go.Scatter(x=z, y=sigmoid_vals, name="Sigmoid", line=dict(color="blue", width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=z, y=relu_vals, name="ReLU", line=dict(color="green", width=3)),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=z, y=tanh_vals, name="Tanh", line=dict(color="red", width=3)),
        row=1,
        col=3,
    )

    # Add zero lines
    for col in [1, 2, 3]:
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=col)
        fig.add_vline(x=0, line_dash="dot", line_color="gray", row=1, col=col)

    fig.update_layout(height=350, showlegend=False, title_text="Common Activation Functions")
    fig.update_xaxes(title_text="Input (z)")
    fig.update_yaxes(title_text="Output", range=[-1, max(relu_vals)] if same_range else None)
    fig.show()


def plot_perceptron_computation(
    weighted_inputs: np.ndarray,
    bias: float,
    z: float,
    output: float,
) -> None:
    """
    Visualizes perceptron computation step-by-step.

    Args:
        weighted_inputs: Array of weighted input values (x_i * w_i).
        bias: Bias term.
        z: Pre-activation value (sum of weighted inputs + bias).
        output: Final output after activation.
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("1. Inputs × Weights", "2. Weighted Sum + Bias", "3. After Activation"),
    )

    n_inputs = len(weighted_inputs)
    input_labels = [f"x{i + 1}×w{i + 1}" for i in range(n_inputs)]

    # Step 1: Weighted inputs
    fig.add_trace(
        go.Bar(
            x=input_labels,
            y=weighted_inputs,
            marker_color=["steelblue" if v >= 0 else "coral" for v in weighted_inputs],
            text=[f"{v:.3f}" for v in weighted_inputs],
            textposition="outside",
        ),
        row=1,
        col=1,
    )

    # Step 2: Sum
    sum_wx = np.sum(weighted_inputs)
    fig.add_trace(
        go.Bar(
            x=["Σ(wx)", "bias", "z"],
            y=[sum_wx, bias, z],
            marker_color=["steelblue", "gray", "purple"],
            text=[f"{sum_wx:.3f}", f"{bias:.3f}", f"{z:.3f}"],
            textposition="outside",
        ),
        row=1,
        col=2,
    )

    # Step 3: Activation
    fig.add_trace(
        go.Bar(
            x=["z", "σ(z)"],
            y=[z, output],
            marker_color=["purple", "green"],
            text=[f"{z:.3f}", f"{output:.3f}"],
            textposition="outside",
        ),
        row=1,
        col=3,
    )

    fig.update_layout(height=400, showlegend=False, title_text="Perceptron Computation Breakdown")
    fig.show()


def plot_layer_response(
    outputs: np.ndarray,
    title: str = "Perceptron Layer Response",
) -> None:
    """
    Visualizes responses from a layer of neurons with bar chart and histogram.

    Args:
        outputs: Array of output values from each neuron.
        title: Plot title.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Individual Outputs", "Distribution"))

    fig.add_trace(go.Bar(y=outputs, marker_color="steelblue"), row=1, col=1)
    fig.add_trace(go.Histogram(x=outputs, nbinsx=10, marker_color="coral"), row=1, col=2)

    fig.add_hline(
        y=outputs.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {outputs.mean():.3f}",
        row=1,
        col=1,
    )

    fig.update_layout(height=400, showlegend=False, title_text=title)
    fig.update_xaxes(title_text="Neuron Index", row=1, col=1)
    fig.update_yaxes(title_text="Output", row=1, col=1)
    fig.update_xaxes(title_text="Output Value", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.show()


def plot_lif_response(
    time: np.ndarray,
    current: np.ndarray,
    v_trace: np.ndarray,
    spike_times: list[float],
    v_threshold: float,
    v_rest: float,
    title: str = "LIF Neuron Response to Step Current",
) -> None:
    """
    Plots LIF neuron response showing input current and membrane potential.

    Args:
        time: Time array (ms).
        current: Input current array (nA).
        v_trace: Membrane potential trace (mV).
        spike_times: List of spike times (ms).
        v_threshold: Firing threshold (mV).
        v_rest: Resting potential (mV).
        title: Plot title.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Input Current", "Membrane Potential"),
        vertical_spacing=0.1,
    )

    fig.add_trace(
        go.Scatter(x=time, y=current, fill="tozeroy", name="I(t)", line=dict(color="coral")),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=time, y=v_trace, name="V(t)", line=dict(color="steelblue")),
        row=2,
        col=1,
    )

    fig.add_hline(
        y=v_threshold, line_dash="dash", line_color="red", annotation_text="Threshold", row=2, col=1
    )
    fig.add_hline(
        y=v_rest, line_dash="dot", line_color="gray", annotation_text="Rest", row=2, col=1
    )

    # Mark spikes
    if spike_times:
        fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=[-40] * len(spike_times),
                mode="markers",
                marker=dict(symbol="star", size=12, color="gold"),
                name="Spikes",
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Current (nA)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (mV)", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)

    fig.update_layout(height=500, title_text=title)
    fig.show()


def plot_fi_curve(
    currents: np.ndarray,
    rates: np.ndarray,
    rheobase: float | None = None,
    title: str = "F-I Curve: How LIF Neurons Encode Input Intensity",
) -> None:
    """
    Plots the F-I curve (firing rate vs input current).

    Args:
        currents: Array of input current values (nA).
        rates: Array of firing rates (Hz).
        rheobase: Optional rheobase value to mark on the plot.
        title: Plot title.
    """
    fig = px.line(
        x=currents,
        y=rates,
        markers=True,
        labels={"x": "Input Current (nA)", "y": "Firing Rate (Hz)"},
        title=title,
    )
    fig.update_traces(line_color="green", marker_color="darkgreen")

    if rheobase is not None:
        fig.add_vline(
            x=rheobase,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Rheobase: {rheobase:.2f} nA",
        )

    fig.update_layout(height=450)
    fig.show()


def plot_synaptic_response(
    time: np.ndarray | list,
    g_trace: np.ndarray | list,
    i_trace: np.ndarray | list,
    spike_times: list[float],
    title: str = "Synaptic Response to Spike Train",
) -> None:
    """
    Plots synaptic conductance and current response to a spike train.

    Args:
        time: Time array (ms).
        g_trace: Conductance trace (nS).
        i_trace: Current trace (nA).
        spike_times: List of presynaptic spike times (ms).
        title: Plot title.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Synaptic Conductance", "Synaptic Current"),
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Scatter(x=time, y=g_trace, fill="tozeroy", name="g_syn", line=dict(color="coral")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time, y=i_trace, fill="tozeroy", name="I_syn", line=dict(color="purple")),
        row=2,
        col=1,
    )

    # Mark spikes
    for row in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=[0] * len(spike_times),
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="red"),
                name="Pre spikes",
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )

    fig.update_yaxes(title_text="g (nS)", row=1, col=1)
    fig.update_yaxes(title_text="I (nA)", row=2, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=2, col=1)

    fig.update_layout(height=500, title_text=title)
    fig.show()


def plot_synapse_comparison(
    time: np.ndarray | list,
    g_regular: np.ndarray | list,
    g_depressing: np.ndarray | list,
    x_trace: np.ndarray | list,
    spike_times: np.ndarray | list,
    title: str = "Stateless vs History-Dependent Synapses",
) -> None:
    """
    Compares regular and depressing synapse responses.

    Args:
        time: Time array (ms).
        g_regular: Conductance trace for regular synapse (nS).
        g_depressing: Conductance trace for depressing synapse (nS).
        x_trace: Vesicle fraction trace for depressing synapse.
        spike_times: Array of presynaptic spike times (ms).
        title: Plot title.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Regular Synapse (Stateless)",
            "Depressing Synapse (History-Dependent)",
            "Available Vesicle Fraction",
        ),
        vertical_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=time, y=g_regular, fill="tozeroy", line=dict(color="coral")), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=g_depressing, fill="tozeroy", line=dict(color="gold")), row=2, col=1
    )
    fig.add_trace(go.Scatter(x=time, y=x_trace, line=dict(color="green", width=2)), row=3, col=1)

    # Mark spikes
    for row in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=[0] * len(spike_times),
                mode="markers",
                marker=dict(symbol="triangle-up", size=8, color="red"),
                showlegend=False,
            ),
            row=row,
            col=1,
        )

    fig.update_yaxes(title_text="g (nS)", row=1, col=1)
    fig.update_yaxes(title_text="g (nS)", row=2, col=1)
    fig.update_yaxes(title_text="x", range=[0, 1.1], row=3, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=3, col=1)

    fig.update_layout(height=650, showlegend=False, title_text=title)
    fig.show()


def plot_stp_comparison(
    time: np.ndarray,
    traces: list[dict],
    spike_times: np.ndarray,
    title: str = "STP Parameter Exploration",
) -> None:
    """
    Plots comparison of multiple STP parameter configurations.

    Args:
        time: Time array (ms).
        traces: List of dicts with 'name', 'g_trace', and 'color' keys.
        spike_times: Array of presynaptic spike times (ms).
        title: Plot title.
    """
    fig = make_subplots(
        rows=len(traces),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[t["name"] for t in traces],
        vertical_spacing=0.05,
    )

    for idx, trace in enumerate(traces):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=trace["g_trace"],
                fill="tozeroy",
                line=dict(color=trace["color"]),
                name=trace["name"],
            ),
            row=idx + 1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=spike_times,
                y=[0] * len(spike_times),
                mode="markers",
                marker=dict(symbol="triangle-up", size=6, color="red"),
                showlegend=False,
            ),
            row=idx + 1,
            col=1,
        )

        fig.update_yaxes(title_text="g (nS)", row=idx + 1, col=1)

    fig.update_xaxes(title_text="Time (ms)", row=len(traces), col=1)
    fig.update_layout(height=700, showlegend=False, title_text=title)
    fig.show()


def plot_complete_neuron(
    time: np.ndarray,
    presynaptic_spikes: np.ndarray,
    g_trace: np.ndarray,
    x_trace: np.ndarray,
    v_trace: np.ndarray,
    spike_times: list[float],
    v_threshold: float,
    title: str = "Complete Biological Neuron: Input → Processing → Output",
) -> None:
    """
    Comprehensive visualization of a complete biological neuron model.

    Args:
        time: Time array (ms).
        presynaptic_spikes: Boolean array indicating presynaptic spike times.
        g_trace: Conductance trace (nS).
        x_trace: Vesicle fraction trace.
        v_trace: Membrane potential trace (mV).
        spike_times: List of postsynaptic spike times (ms).
        v_threshold: Firing threshold (mV).
        title: Plot title.
    """
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Presynaptic Spikes (INPUT)",
            "Conductance & Vesicles",
            "Membrane Potential (PROCESSING)",
            "Postsynaptic Spikes (OUTPUT)",
        ),
        vertical_spacing=0.06,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    # Presynaptic spikes
    pre_spike_times = time[presynaptic_spikes]
    fig.add_trace(
        go.Scatter(
            x=pre_spike_times,
            y=[1] * len(pre_spike_times),
            mode="markers",
            marker=dict(symbol="line-ns-open", size=6, color="blue"),
            name="Pre",
        ),
        row=1,
        col=1,
    )

    # Conductance and vesicles
    fig.add_trace(
        go.Scatter(x=time, y=g_trace, name="g", line=dict(color="coral")),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=time, y=x_trace, name="x", line=dict(color="green", dash="dash")),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Membrane potential
    fig.add_trace(
        go.Scatter(x=time, y=v_trace, name="V", line=dict(color="steelblue")),
        row=3,
        col=1,
    )
    fig.add_hline(y=v_threshold, line_dash="dash", line_color="red", row=3, col=1)

    # Postsynaptic spikes
    fig.add_trace(
        go.Scatter(
            x=spike_times,
            y=[1] * len(spike_times),
            mode="markers",
            marker=dict(symbol="star", size=10, color="gold"),
            name="Post",
        ),
        row=4,
        col=1,
    )

    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="g (nS)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="x", range=[0, 1.1], row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="V (mV)", row=3, col=1)
    fig.update_yaxes(showticklabels=False, row=4, col=1)
    fig.update_xaxes(title_text="Time (ms)", row=4, col=1)

    fig.update_layout(height=750, title_text=title)
    fig.show()


def plot_io_comparison(
    perceptron_inputs: np.ndarray,
    perceptron_outputs: np.ndarray,
    lif_inputs: np.ndarray,
    lif_outputs: np.ndarray,
    title: str = "Input-Output Comparison",
) -> None:
    """
    Compares I-O curves for perceptron and biological neuron.

    Args:
        perceptron_inputs: Input values for perceptron.
        perceptron_outputs: Output activations from perceptron.
        lif_inputs: Input rates (Hz) for LIF neuron.
        lif_outputs: Output rates (Hz) from LIF neuron.
        title: Plot title.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Perceptron", "Biological Neuron"))

    fig.add_trace(
        go.Scatter(
            x=perceptron_inputs,
            y=perceptron_outputs,
            mode="lines+markers",
            line=dict(color="steelblue", width=3),
            marker=dict(size=8),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=lif_inputs,
            y=lif_outputs,
            mode="lines+markers",
            line=dict(color="green", width=3),
            marker=dict(size=8),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Input Value", row=1, col=1)
    fig.update_yaxes(title_text="Output (activation)", row=1, col=1)
    fig.update_xaxes(title_text="Input Rate (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Output Rate (Hz)", row=1, col=2)

    fig.update_layout(height=400, showlegend=False, title_text=title)
    fig.show()


# =============================================================================
# Motor Control Tutorial Plotting Functions
# =============================================================================


def plot_reaching_trajectories(
    trajectories: list[tuple[np.ndarray, np.ndarray, str, str]],
    target: float,
    title: str = "Reach Trajectories under Signal-Dependent Noise",
) -> None:
    """
    Plots position trajectories and control signals for reaching movements.

    Args:
        trajectories: List of tuples (traj, controls, label, color)
            - traj: array of shape (steps, 2) with [position, velocity]
            - controls: array of shape (steps,) with control signals
            - label: string label for the trajectory
            - color: color for the plot
        target: Target position
        title: Plot title
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Position Trajectories", "Control Signals"),
        vertical_spacing=0.12,
    )

    for traj, controls, label, color in trajectories:
        steps = len(traj)
        time = np.arange(steps) * 0.01

        # Plot position
        fig.add_trace(
            go.Scatter(x=time, y=traj[:, 0], name=label, line=dict(color=color), showlegend=True),
            row=1,
            col=1,
        )

        # Plot control signals
        fig.add_trace(
            go.Scatter(x=time, y=controls, name=label, line=dict(color=color), showlegend=False),
            row=2,
            col=1,
        )

    # Add target line
    fig.add_hline(
        y=target, line_dash="dash", line_color="green", annotation_text="Target", row=1, col=1
    )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Force (u)", row=2, col=1)

    fig.update_layout(height=600, title_text=title)
    fig.show()


def plot_kinematic_profile(
    time: np.ndarray,
    traj: np.ndarray,
    target: float,
    title: str = "Kinematic Profile of Reaching Movement",
) -> None:
    """
    Plots position, velocity, and acceleration profiles of a reaching movement.

    Args:
        time: Time array (seconds)
        traj: Trajectory array of shape (steps, 2) with [position, velocity]
        target: Target position
        title: Plot title
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Position", "Velocity (Speed)", "Acceleration"),
        vertical_spacing=0.1,
    )

    # Position
    fig.add_trace(
        go.Scatter(x=time, y=traj[:, 0], name="Position", line=dict(color="blue")),
        row=1,
        col=1,
    )
    fig.add_hline(y=target, line_dash="dash", line_color="green", row=1, col=1)

    # Velocity
    fig.add_trace(
        go.Scatter(x=time, y=traj[:, 1], name="Velocity", line=dict(color="red")),
        row=2,
        col=1,
    )

    # Acceleration (computed from velocity)
    accel = np.gradient(traj[:, 1], time)
    fig.add_trace(
        go.Scatter(x=time, y=accel, name="Acceleration", line=dict(color="purple")),
        row=3,
        col=1,
    )

    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Position", row=1, col=1)
    fig.update_yaxes(title_text="Velocity", row=2, col=1)
    fig.update_yaxes(title_text="Accel.", row=3, col=1)

    fig.update_layout(height=700, title_text=title, showlegend=False)
    fig.show()


def plot_fitts_law(
    index_of_difficulty: list[np.ndarray],
    movement_times: list[np.ndarray],
    labels: list[str],
    title: str = "Fitts' Law: Movement Time vs Index of Difficulty",
) -> None:
    """
    Plots Fitts' Law relationship between movement time and task difficulty.

    Args:
        index_of_difficulty: List of arrays with index of difficulty values
        movement_times: List of arrays with corresponding movement times
        labels: List of labels for each condition
        title: Plot title
    """
    fig = go.Figure()

    for ID, MT, label in zip(index_of_difficulty, movement_times, labels):
        fig.add_trace(go.Scatter(x=ID, y=MT, mode="lines+markers", name=label, marker=dict(size=8)))

    fig.update_layout(
        title=title,
        xaxis_title="Index of Difficulty (bits)",
        yaxis_title="Movement Time (s)",
        height=500,
    )
    fig.show()


def plot_controller_comparison(
    time: np.ndarray,
    position: np.ndarray,
    target: float,
    kp: float,
    kd: float,
    metrics: dict,
    title: str = "Controller Performance",
) -> None:
    """
    Plots controller performance with metrics displayed.

    Args:
        time: Time array (seconds)
        position: Position trajectory
        target: Target position
        kp: Proportional gain
        kd: Derivative gain
        metrics: Dictionary with 'final_error', 'max_overshoot', 'settling_time'
        title: Plot title
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time, y=position, name="Position", line=dict(color="blue")))
    fig.add_hline(y=target, line_dash="dash", line_color="green", annotation_text="Target")

    # Add metrics as annotation
    metrics_text = (
        f"Kp={kp}, Kd={kd}<br>"
        f"Final Error: {metrics['final_error']:.4f}<br>"
        f"Max Overshoot: {metrics['max_overshoot']:.4f}<br>"
        f"Settling Time: {metrics['settling_time']:.3f}s"
        if metrics["settling_time"]
        else "Did not settle"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=metrics_text,
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        xanchor="left",
        yanchor="top",
    )

    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Position", height=400)
    fig.show()


def plot_endpoint_variance(
    noise_levels: np.ndarray,
    variances: np.ndarray,
    title: str = "Endpoint Variance vs Noise Level",
) -> None:
    """
    Plots how endpoint position variance scales with noise level.

    Args:
        noise_levels: Array of noise scaling factors (alpha)
        variances: Array of corresponding standard deviations
        title: Plot title
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=noise_levels,
            y=variances,
            mode="lines+markers",
            marker=dict(size=10, color="red"),
            line=dict(width=3, color="red"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Noise Level (α)",
        yaxis_title="Endpoint Std Dev",
        height=450,
    )
    fig.show()


def plot_energy_analysis(
    gains: list[tuple[float, float]],
    energies: list[float],
    movement_times: list[float],
    final_errors: list[float],
) -> None:
    """
    Plots energy-time-accuracy tradeoffs for different controller gains.

    Args:
        gains: List of (kp, kd) tuples
        energies: List of total energy costs
        movement_times: List of movement times
        final_errors: List of final position errors
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Energy Cost", "Movement Time", "Final Error"),
    )

    gain_labels = [f"({kp:.1f}, {kd:.1f})" for kp, kd in gains]

    # Energy
    fig.add_trace(
        go.Bar(x=gain_labels, y=energies, marker_color="steelblue"),
        row=1,
        col=1,
    )

    # Movement time
    fig.add_trace(
        go.Bar(x=gain_labels, y=movement_times, marker_color="coral"),
        row=1,
        col=2,
    )

    # Final error
    fig.add_trace(
        go.Bar(x=gain_labels, y=final_errors, marker_color="green"),
        row=1,
        col=3,
    )

    fig.update_xaxes(title_text="(Kp, Kd)", row=1, col=1)
    fig.update_xaxes(title_text="(Kp, Kd)", row=1, col=2)
    fig.update_xaxes(title_text="(Kp, Kd)", row=1, col=3)

    fig.update_yaxes(title_text="Energy", row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Error", row=1, col=3)

    fig.update_layout(
        height=400,
        title_text="Controller Tradeoffs: Energy, Speed, and Accuracy",
        showlegend=False,
    )
    fig.show()
