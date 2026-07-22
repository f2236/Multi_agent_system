"""
Channel model for UAV-to-antenna communication.

Implements wireless link computation: path loss, SNR, data rate.
"""

import math
from typing import Tuple
from uav_optimizer.domain import Position3D, AntennaConfig, ChannelState


class ChannelModel:
    """Wireless channel model for free-space propagation."""

    # Boltzmann constant (J/K)
    BOLTZMANN_CONSTANT = 1.38e-23
    # Reference temperature (Kelvin)
    REFERENCE_TEMPERATURE = 290.0

    @staticmethod
    def free_space_path_loss(
        distance: float, frequency_hz: float = 2.4e9
    ) -> float:
        """
        Compute free-space path loss in linear scale.

        Args:
            distance: Distance in meters (must be > 0)
            frequency_hz: Frequency in Hz

        Returns:
            Path loss as linear ratio (watts received / watts transmitted)

        Raises:
            ValueError: If distance <= 0
        """
        if distance <= 0:
            raise ValueError(f"Distance must be positive, got {distance}")

        # Speed of light
        c = 3e8
        wavelength = c / frequency_hz
        # Friis transmission equation: PL = (4πd/λ)²
        path_loss_db = 20 * math.log10(4 * math.pi * distance / wavelength)
        # Convert to linear
        return 10 ** (path_loss_db / 10)

    @staticmethod
    def compute_noise_power(
        bandwidth_hz: float,
        noise_figure_db: float,
        temperature_k: float = REFERENCE_TEMPERATURE,
    ) -> float:
        """
        Compute thermal noise power in watts.

        Args:
            bandwidth_hz: Bandwidth in Hz
            noise_figure_db: Noise figure in dB
            temperature_k: Temperature in Kelvin

        Returns:
            Noise power in watts
        """
        # Thermal noise power: P_n = k * T * B
        thermal_noise = (
            ChannelModel.BOLTZMANN_CONSTANT * temperature_k * bandwidth_hz
        )
        # Apply noise figure: P_n_total = P_n * NF
        noise_figure_linear = 10 ** (noise_figure_db / 10)
        return thermal_noise * noise_figure_linear

    @staticmethod
    def compute_snr(
        tx_power_w: float,
        distance: float,
        bandwidth_hz: float,
        noise_figure_db: float,
        frequency_hz: float = 2.4e9,
    ) -> float:
        """
        Compute signal-to-noise ratio.

        Args:
            tx_power_w: Transmit power in watts
            distance: Distance in meters
            bandwidth_hz: Bandwidth in Hz
            noise_figure_db: Noise figure in dB
            frequency_hz: Frequency in Hz

        Returns:
            SNR in linear scale
        """
        path_loss = ChannelModel.free_space_path_loss(distance, frequency_hz)
        noise_power = ChannelModel.compute_noise_power(bandwidth_hz, noise_figure_db)

        # SNR = (P_tx / PL) / P_n
        if noise_power <= 0:
            return float("inf")
        return (tx_power_w / path_loss) / noise_power

    @staticmethod
    def compute_data_rate(
        tx_power_w: float,
        distance: float,
        bandwidth_hz: float,
        noise_figure_db: float,
        frequency_hz: float = 2.4e9,
    ) -> float:
        """
        Compute link data rate using Shannon capacity.

        R = B * log2(1 + SNR)

        Args:
            tx_power_w: Transmit power in watts
            distance: Distance in meters
            bandwidth_hz: Bandwidth in Hz
            noise_figure_db: Noise figure in dB
            frequency_hz: Frequency in Hz

        Returns:
            Data rate in bits per second
        """
        snr = ChannelModel.compute_snr(
            tx_power_w, distance, bandwidth_hz, noise_figure_db, frequency_hz
        )
        # Shannon capacity: R = B * log2(1 + SNR)
        if snr < 0:
            return 0.0
        return bandwidth_hz * math.log2(1.0 + snr)


class LinkEvaluator:
    """Evaluates individual UAV-to-antenna links."""

    @staticmethod
    def evaluate_link(
        uav_position: Position3D,
        antenna: AntennaConfig,
    ) -> ChannelState:
        """
        Evaluate a single UAV-antenna link.

        Args:
            uav_position: UAV 3D position
            antenna: Antenna configuration

        Returns:
            ChannelState with link metrics
        """
        # Compute distance
        dx = uav_position.x - antenna.position.x
        dy = uav_position.y - antenna.position.y
        dz = uav_position.z - antenna.position.z
        distance = math.sqrt(dx**2 + dy**2 + dz**2)

        # Clamp distance to avoid singularities
        distance = max(distance, 1.0)

        # Convert tx power from dBm to watts
        tx_power_w = 10 ** ((antenna.tx_power_dbm - 30) / 10)

        # Compute path loss in dB
        path_loss_linear = ChannelModel.free_space_path_loss(
            distance, antenna.frequency_hz
        )
        path_loss_db = 10 * math.log10(path_loss_linear)

        # Compute SNR in dB
        snr_linear = ChannelModel.compute_snr(
            tx_power_w,
            distance,
            antenna.bandwidth_hz,
            antenna.noise_figure_db,
            antenna.frequency_hz,
        )
        snr_db = 10 * math.log10(max(snr_linear, 1e-10))

        # Compute data rate
        data_rate_bps = ChannelModel.compute_data_rate(
            tx_power_w,
            distance,
            antenna.bandwidth_hz,
            antenna.noise_figure_db,
            antenna.frequency_hz,
        )

        return ChannelState(
            uav_id=0,  # Will be set by caller
            antenna_id=antenna.antenna_id,
            distance=distance,
            path_loss_db=path_loss_db,
            snr_db=snr_db,
            data_rate_bps=data_rate_bps,
        )
