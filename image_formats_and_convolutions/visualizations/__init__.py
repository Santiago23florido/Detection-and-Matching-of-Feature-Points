"""Visualization submodule — one plot function per file.

All plot functions follow the same signature convention:
    plot_<name>(..., output_path: Path) -> None
"""

from .timing_comparison import plot_timing_comparison
from .timing_boxplot import plot_timing_boxplot
from .time_vs_image_size import plot_time_vs_image_size
from .visual_comparison import plot_visual_comparison
from .kernel_decomposition import plot_kernel_decomposition
from .unsharp_components import plot_component_images
from .profile_1d import plot_1d_profile_analysis
from .histogram_comparison import plot_histogram_comparison
from .gradient_components import plot_gradient_components
from .gradient_display_precautions import plot_display_precautions
from .gradient_kernels import plot_gradient_kernels
from .gradient_1d_profile import plot_gradient_1d_profile
from ._colors import METHOD_COLORS

__all__ = [
    "plot_timing_comparison",
    "plot_timing_boxplot",
    "plot_time_vs_image_size",
    "plot_visual_comparison",
    "plot_kernel_decomposition",
    "plot_component_images",
    "plot_1d_profile_analysis",
    "plot_histogram_comparison",
    "plot_gradient_components",
    "plot_display_precautions",
    "plot_gradient_kernels",
    "plot_gradient_1d_profile",
    "METHOD_COLORS",
]
