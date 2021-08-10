# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import numpy as np
import scipy

class RandomKSpaceLayer:
    """
    generate random k space augmentations
    """

    def __init__(self, name='random_kspace', alpha=1.0):
        self.name = name
        self.acquisition_type = '2D'
        self.alpha = alpha

        self._apply_kspace_transform = True
        self._apply_highpass = False
        self._apply_lowpass = True
        self._apply_scan_mask = False
        self._apply_rf_spike = False
        self._apply_noise = False
        self._apply_wrap = False
        self._apply_phase_shift = False

        # Augmentation probabilities
        self.highpass_prob = 0.0
        self.lowpass_prob = alpha
        self.scan_mask_prob = 0.0
        self.rf_prob = 0.0
        self.noise_prob = 0.0
        self.wrap_prob = 0.0
        self.phase_shift_prob = 0.0

        # High-pass filter
        self.highpass = 0.0
        self.max_highpass = 1.0
        self.min_highpass = 0.0
        self.highpass_type = 'rect'  # 'radial' or 'rect'
        self.highpass_axis = None

        # Low-pass filter
        self.lowpass = 0.0
        self.max_lowpass = 1.0
        self.min_lowpass = 0.1
        self.lowpass_type = 'rect'  # 'radial' or 'rect'
        self.lowpass_axis = 0

        # Mask scan
        self.scan_percentage = 0.0
        self.min_scan = 0.0
        self.max_scan = 1.0

        # SNR
        self.snr = 10.0  # db
        self.min_snr = 5.0
        self.max_snr = 30.0

        # RF spike
        self.rf_strength = 1.0
        self.min_rf_strength = 1.0
        self.max_rf_strength = 10.0

        # Wrap
        self.wrap_axis = -1  # 0,1,2 or -1 for random
        self.wrap_spacing = 2

        # Phase shift
        self.shift_axis = -1  # 0,1,2 or -1 for random
        self.shift = None
        self.min_shift = -10.0  # voxels
        self.max_shift = 10.0
        self.shift_lines = 0  # number of shifted kspace lines
        self.min_lines = 0
        self.max_lines = 100

        # Mean filter

    def init_kspace_transform(self, kspace_augmentation=False):
        self._apply_kspace_transform = kspace_augmentation

    def init_highpass_prob(self, highpass_prob=0.0):
        self.highpass_prob = float(highpass_prob)

    def init_lowpass_prob(self, lowpass_prob=0.0):
        self.lowpass_prob = float(lowpass_prob)

    def init_scan_mask_prob(self, scan_mask_prob=0.0):
        self.scan_mask_prob = float(scan_mask_prob)

    def init_rf_prob(self, rf_prob=0.0):
        self.rf_prob = float(rf_prob)

    def init_noise_prob(self, noise_prob=0.0):
        self.noise_prob = float(noise_prob)

    def init_wrap_prob(self, wrap_prob=0.0):
        self.wrap_prob = float(wrap_prob)

    def init_phase_shift_prob(self, phase_shift_prob=0.0):
        self.phase_shift_prob = float(phase_shift_prob)

    def init_highpass(self, highpass_range=(0.0, 0.5), highpass_axis=None):
        assert highpass_range[0] < highpass_range[1]
        self.min_highpass = float(highpass_range[0])
        self.max_highpass = float(highpass_range[1])
        self.highpass_axis = highpass_axis
        print('min_highpass:', self.min_highpass, 'max_highpass:', self.max_highpass, 'axis:', self.highpass_axis)

    def init_lowpass(self, lowpass_range=(0.0, 0.5), lowpass_axis=None):
        assert lowpass_range[0] < lowpass_range[1]
        self.min_lowpass = float(lowpass_range[0])
        self.max_lowpass = float(lowpass_range[1])
        self.lowpass_axis = lowpass_axis

    #        print('min_lowpass:', self.min_lowpass, 'max_lowpass:', self.max_lowpass, 'axis:', self.lowpass_axis)

    def init_scan_mask_range(self, scan_mask_range=(0.0, 0.5)):
        assert scan_mask_range[0] < scan_mask_range[1]
        self.min_scan = float(scan_mask_range[0])
        self.max_scan = float(scan_mask_range[1])
        print('min_scan:', self.min_scan, 'max_scan', self.max_scan)

    def init_rf_range(self, rf_range=(1.0, 100.0)):
        assert rf_range[0] < rf_range[1]
        self.min_rf_strength = float(rf_range[0])
        self.max_rf_strength = float(rf_range[1])
        print('min_rf:', self.min_rf_strength, 'max_rf:', self.max_rf_strength)

    def init_snr_range(self, snr_range=(5.0, 30.0)):
        assert snr_range[0] < snr_range[1]
        self.min_snr = float(snr_range[0])
        self.max_snr = float(snr_range[1])
        print('min_snr:', self.min_snr, 'max_snr:', self.max_snr)

    def init_wrap(self, wrap_axis=-1):
        self.wrap_axis = int(wrap_axis)
        print('wrap_axis:', self.wrap_axis)

    def init_phase_shift(self, shift_axis=-1, shift_range=(-10.0, 10.0), shift_lines_range=(0, 100)):
        assert shift_range[0] < shift_range[1]
        assert shift_lines_range[0] < shift_lines_range[1]
        self.shift_axis = int(shift_axis)
        self.shift = np.array([[0.0, 0.0, 0.0]])
        self.min_shift = float(shift_range[0])
        self.max_shift = float(shift_range[1])
        self.min_lines = int(shift_lines_range[0])
        self.max_lines = int(shift_lines_range[1])

        print('shift_axis:', self.shift_axis, 'min_shift:', self.min_shift, 'max_shift:', self.max_shift, 'min_lines:',
              self.min_lines, 'max_lines:', self.max_lines)

    def computeFourierTransform(self, image):
        F = np.zeros(image.shape, np.complex)
        if self.acquisition_type == '3D':
            F = np.fft.fftshift(np.fft.fftn(image))
        elif self.acquisition_type == '2D':
            #            print('got here')
            # for k in range(image_3d.shape[2]):
            Ik = image
            F = np.fft.fftshift(np.fft.fft2(Ik))
        return F

    def computeInverseFourierTransform(self, F):
        IF = np.zeros(F.shape, np.complex)
        if self.acquisition_type == '3D':
            IF = np.fft.ifftn(np.fft.ifftshift(F))
        elif self.acquisition_type == '2D':
            #            print('got into inverse')
            Fk = F
            IF = np.fft.ifft2(np.fft.ifftshift(Fk))
        return IF

    def addComplexNoise(self, F, snr):
        rows, cols, depth = F.shape
        signalPower_lin = np.sum(np.conj(F) * F) / float(F.size)
        signalPower_dB = 10.0 * np.log10(signalPower_lin)
        noisePower_dB = signalPower_dB - snr  # do 30dB SNR
        noisePower_lin = 10.0 ** (noisePower_dB / 10.0)
        noise = np.sqrt(noisePower_lin / 2.0) * (
                np.random.randn(rows, cols, depth) + 1j * np.random.randn(rows, cols, depth))
        return F + noise

    def maskCircle(self, F, radius):
        rows, cols, depth = F.shape
        c_x, c_y, c_z = (0.5 * cols, 0.5 * rows, 0.5 * depth)
        xx = np.linspace(0, cols - 1, cols) + 0.5
        yy = np.linspace(0, rows - 1, rows) + 0.5
        zz = np.linspace(0, depth - 1, depth) + 0.5
        X, Y, Z = np.meshgrid(xx, yy, zz)
        dists_sqrd = (X - c_x) ** 2 + (Y - c_y) ** 2 + (Z - c_z) ** 2
        return dists_sqrd < radius ** 2

    def maskRect(self, F, ratio, axis=None):
        rows, cols = F.shape
        c_x, c_y = (0.5 * cols, 0.5 * rows)
        xx = np.linspace(0, cols - 1, cols) + 0.5
        yy = np.linspace(0, rows - 1, rows) + 0.5
        X, Y = np.meshgrid(xx, yy)
        dists_x = np.abs(X - c_x)
        dists_y = np.abs(Y - c_y)

        if axis == 0:
            radius = np.ceil(cols * ratio)
            return dists_x < radius
        elif axis == 1:
            radius = np.ceil(rows * ratio)
            return dists_y < radius

    def highpassFilter(self, F, ratio, axis):
        rows, cols, depth = F.shape
        radius = np.ceil((np.max((rows * ratio, cols * ratio, depth * ratio)) / 2.0))
        if self.highpass_type == 'radial':
            mask = self.maskCircle(F, radius)
        else:
            mask = self.maskRect(F, radius, axis)
        F = F * (1 - mask)
        return F

    def lowpassFilter(self, F, ratio, axis):
        mask = self.maskRect(F, ratio, axis)
        mask = scipy.ndimage.gaussian_filter1d(mask.astype(np.float32), sigma=100, axis=axis)
        F = F * mask
        return F

    def maskScan(self, F, ratio):
        rows, cols, depth = F.shape
        d = int(depth * ratio / 2.0)
        F[:, :, 0:d] = 0.0
        F[:, :, depth - d:depth] = 0.0
        return F

    def positive_or_negative(self):
        return 1.0 if np.random.random() < 0.5 else -1.0

    def rfspike(self, F, strength):
        N = F.size
        ind = np.random.randint(N)
        r, c, d = np.unravel_index(ind, F.shape)
        if strength >= 0:
            F[r, c, d] = (F[r, c, d] / np.abs(F[r, c, d])) * np.max((np.abs(F.max()) * strength, np.abs(F[r, c, d])))
        else:
            F[r, c, d] = (F[r, c, d] / np.abs(F[r, c, d])) * np.min((np.abs(F.max()) * strength, -np.abs(F[r, c, d])))
        # F[r, c, d] *= strength
        return F

    def wrap(self, F, axis, s):
        if axis == 0:
            F[::s, :, :] = 0.0
        elif axis == 1:
            F[:, ::s, :] = 0.0
        elif axis == 2:
            F[:, :, ::s] = 0.0
        return F

    def phaseShift(self, F, trans, num_lines):
        rows, cols, depth = F.shape
        xx = np.linspace(-0.5, 0.5, cols)
        yy = np.linspace(-0.5, 0.5, rows)
        zz = np.linspace(-0.5, 0.5, depth)
        X, Y, Z = np.meshgrid(xx, yy, zz)
        k = np.stack((X, Y, Z), axis=3)

        # Phase shift
        # Ft = F * np.exp(1j * 2.0 * np.pi * np.sum(k * trans, axis=3))

        inds = np.random.randint(0, depth, num_lines)
        Ft = np.copy(F)
        for i in range(num_lines):
            ind = inds[i]
            Ft[:, :, ind] = F[:, :, ind] * np.exp(1j * 2.0 * np.pi * np.sum(k[:, :, ind, :] * trans[i, :], axis=2))

        ratio = 0.07
        radius = int(np.max((rows * ratio, cols * ratio, depth * ratio)) / 2.0)
        mask = self.maskCircle(F, radius)
        F = F * mask + Ft * (1 - mask)
        return F

    def randomise(self, spatial_rank=3):
        if self._apply_kspace_transform:
            if np.random.random_sample() < self.highpass_prob:
                self._apply_highpass = True
                if spatial_rank == 3:
                    self._randomise_highpass()
                else:
                    pass
            else:
                self._apply_highpass = False
                # print('No highpass filter')

            if np.random.random_sample() < self.lowpass_prob:
                self._apply_lowpass = True
                #                print('spatial rank', spatial_rank)
                if spatial_rank == 3:
                    self._randomise_lowpass()
                elif spatial_rank == 2:
                    self._randomise_lowpass()
                else:
                    raise NotImplementedError
            else:
                self._apply_lowpass = False
                # print('No lowpass filter')

            if np.random.random_sample() < self.scan_mask_prob:
                self._apply_scan_mask = True
                if spatial_rank == 3:
                    self._randomise_scan_mask()
                else:
                    pass
            else:
                self._apply_scan_mask = False
                # print('No scan mask')

            if np.random.random_sample() < self.noise_prob:
                self._apply_noise = True
                if spatial_rank == 3:
                    self._randomise_snr()
                else:
                    pass
            else:
                self._apply_noise = False
                # print('No noise augmentation')

            if np.random.random_sample() < self.rf_prob:
                self._apply_rf_spike = True
                if spatial_rank == 3:
                    self._randomise_rf()
                else:
                    pass
            else:
                self._apply_rf_spike = False
                # print('No rf spike augmentation')

            if np.random.random_sample() < self.wrap_prob:
                self._apply_wrap = True
                if spatial_rank == 3:
                    self._randomise_wrap()
                else:
                    pass
            else:
                self._apply_wrap = False
                # print('No wrap augmentation')

            if np.random.random_sample() < self.phase_shift_prob:
                self._apply_phase_shift = True
                if spatial_rank == 3:
                    self._randomise_phase_shift()
                else:
                    pass
            else:
                self._apply_phase_shift = False
                # print('No phase shift augmentation')

            # Make augmentations exclusive
            if self._apply_noise and self._apply_rf_spike:
                if np.random.random_sample() < 0.5:
                    self._apply_rf_spike = False
                else:
                    self._apply_noise = False

            if self._apply_highpass and self._apply_lowpass:
                if np.random.random_sample() < 0.5:
                    self._apply_lowpass = False
                else:
                    self._apply_highpass = False

    def _randomise_highpass(self):
        self.highpass = np.random.uniform(self.min_highpass, self.max_highpass)
        print('highpass:', self.highpass)

    def _randomise_lowpass(self):
        self.lowpass = 1.0 / 2 ** (np.random.randint(0, 4))
        self.lowpass_axis = np.random.randint(0, 2)

    #        print('lowpass:', self.lowpass, 'lowpass axis:', self.lowpass_axis)

    def _randomise_scan_mask(self):
        self.scan_percentage = np.random.uniform(self.min_scan, self.max_scan)
        print('scan_percentage:', self.scan_percentage)

    def _randomise_snr(self):
        self.snr = np.random.uniform(self.min_snr, self.max_snr)
        print('snr:', self.snr)

    def _randomise_rf(self):
        self.rf_strength = np.random.uniform(self.min_rf_strength, self.max_rf_strength)
        self.rf_strength *= self.positive_or_negative()
        print('rf_strength:', self.rf_strength)

    def _randomise_wrap(self):
        if self.wrap_axis == -1:
            self.wrap_axis = np.random.randint(0, 3)
        self.wrap_spacing = np.random.randint(2, 10)
        print('wrap_axis:', self.wrap_axis, 'spacing:', self.wrap_spacing)

    def _randomise_phase_shift(self):
        self.shift_lines = np.random.randint(self.min_lines, self.max_lines)
        if self.shift_axis == -1:
            self.shift = np.random.uniform(self.min_shift, self.max_shift, (self.shift_lines, 3))
        else:
            amount = np.random.uniform(self.min_shift, self.max_shift, (self.shift_lines, 1))
            self.shift = np.zeros((self.shift_lines, 3))
            self.shift[:, self.shift_axis] = amount
        print('shift:', self.shift.shape, 'shift_lines:', self.shift_lines)

    def _transform_kspace(self, image):
        has_kspace_augmentation = np.zeros(4)
        if self._apply_kspace_transform == True:
            F = self.computeFourierTransform(image)

            if self._apply_highpass == True:
                print('Applying highpass filter')
                F = self.highpassFilter(F, self.highpass, self.highpass_axis)
                has_kspace_augmentation[0] = 1

            if self._apply_lowpass == True:
                #                print('Applying lowpass filter')
                F = self.lowpassFilter(F, self.lowpass, self.lowpass_axis)
                has_kspace_augmentation[1] = 1

            if self._apply_noise == True:
                print('Applying noise')
                F = self.addComplexNoise(F, self.snr)
                has_kspace_augmentation[2] = 1

            if self._apply_rf_spike == True:
                print('Applying rf spike')
                F = self.rfspike(F, self.rf_strength)
                has_kspace_augmentation[3] = 1

            if self._apply_scan_mask == True:
                print('Applying scan mask')
                F = self.maskScan(F, self.scan_percentage)
                # has_kspace_augmentation = 1

            if self._apply_wrap == True:
                print('Applying wrap')
                F = self.wrap(F, self.wrap_axis, self.wrap_spacing)
                # has_kspace_augmentation = 1

            if self._apply_phase_shift == True:
                print('Applying phase shift')
                F = self.phaseShift(F, self.shift, self.shift_lines)
                # has_kspace_augmentation = 1

            IF = self.computeInverseFourierTransform(F)
            image = np.real(IF).astype(np.float32)

        return image, has_kspace_augmentation

    def _apply_transformation(self, image):
        """
        :param image: image on which to apply kspace augmentation
        :return: modified image
        """
        self.randomise(spatial_rank=len(image.shape))
        ks_image, has_kspace_augmentation = self._transform_kspace(image)
        #        print('has_kspace_augmentation', has_kspace_augmentation)
        return ks_image, has_kspace_augmentation

    def layer_op(self, inputs, interp_orders, train_on=False, *args, **kwargs):
        if inputs is None:
            return inputs
        for mod_i in range(inputs.shape[0]):
            min_val = np.min(inputs[mod_i, ...])
            mask = np.where(inputs[mod_i, ...] == min_val,
                            np.zeros_like(inputs[mod_i, ...]),
                            np.ones_like(inputs[mod_i, ...]))
            if len(inputs.shape) == 3:
                #                print('inputs[:, mod_i, ...].shape', inputs[mod_i, ...].shape)
                inputs[mod_i, ...], has_kspace_augmentation = \
                    self._apply_transformation(inputs[mod_i, ...])
                inputs[mod_i, ...] = np.where(
                    inputs[mod_i, ...] * mask == 0,
                    np.ones_like(mask) * min_val,
                    inputs[mod_i, ...] * mask)
            else:
                raise NotImplementedError("unknown input format")
        return inputs
