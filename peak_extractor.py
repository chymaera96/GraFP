import numpy as np
import os
import librosa
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

def locmax(vec, indices=False):
    """ Return a boolean vector of which points in vec are local maxima.
        End points are peaks if larger than single neighbors.
        if indices=True, return the indices of the True values instead
        of the boolean vector.
    """
    # vec[-1]-1 means last value can be a peak
    # nbr = np.greater_equal(np.r_[vec, vec[-1]-1], np.r_[vec[0], vec])
    # the np.r_ was killing us, so try an optimization...
    nbr = np.zeros(len(vec) + 1, dtype=bool)
    nbr[0] = True
    nbr[1:-1] = np.greater_equal(vec[1:], vec[:-1])
    maxmask = (nbr[:-1] & ~nbr[1:])
    if indices:
        return np.nonzero(maxmask)[0]
    else:
        return maxmask


class Analyzer(object):
    """ A class to wrap up all the parameters associated with
        the analysis of soundfiles into fingerprints """
    # Parameters

    # optimization: cache pre-calculated Gaussian profile
    __sp_width = None
    __sp_len = None
    __sp_vals = []

    def __init__(self, cfg, density=None):
        if density is None:
            self.density = cfg['density']
        else:
            self.density = density
        self.target_sr = cfg['fs']
        self.n_fft = cfg['n_fft']
        self.n_hop = cfg['hop_len']
        # how wide to spreak peaks
        self.f_sd = cfg['f_sd']
        # Maximum number of local maxima to keep per frame
        self.maxpksperframe = cfg['maxpksperframe']
        self.oversamp = cfg['oversamp']
        self.hpf_pole = cfg['hpf_pole']


    def spreadpeaksinvector(self, vector, width=4.0):
        """ Create a blurred version of vector, where each of the local maxes
            is spread by a gaussian with SD <width>.
        """
        npts = len(vector)
        peaks = locmax(vector, indices=True)
        return self.spreadpeaks(zip(peaks, vector[peaks]),
                                npoints=npts, width=width)

    def spreadpeaks(self, peaks, npoints=None, width=4.0, base=None):
        """ Generate a vector consisting of the max of a set of Gaussian bumps
        :params:
          peaks : list
            list of (index, value) pairs giving the center point and height
            of each gaussian
          npoints : int
            the length of the output vector (needed if base not provided)
          width : float
            the half-width of the Gaussians to lay down at each point
          base : np.array
            optional initial lower bound to place Gaussians above
        :returns:
          vector : np.array(npoints)
            the maximum across all the scaled Gaussians
        """
        if base is None:
            vec = np.zeros(npoints)
        else:
            npoints = len(base)
            vec = np.copy(base)
        # binvals = np.arange(len(vec))
        # for pos, val in peaks:
        #   vec = np.maximum(vec, val*np.exp(-0.5*(((binvals - pos)
        #                                /float(width))**2)))
        if width != self.__sp_width or npoints != self.__sp_len:
            # Need to calculate new vector
            self.__sp_width = width
            self.__sp_len = npoints
            self.__sp_vals = np.exp(-0.5 * ((np.arange(-npoints, npoints + 1)
                                             / width)**2))
        # Now the actual function
        for pos, val in peaks:
            vec = np.maximum(vec, val * self.__sp_vals[np.arange(npoints)
                                                       + npoints - pos])
        return vec

    def _decaying_threshold_fwd_prune(self, sgram, a_dec):
        """ forward pass of findpeaks
            initial threshold envelope based on peaks in first 10 frames
        """
        (srows, scols) = np.shape(sgram)
        sthresh = self.spreadpeaksinvector(
            np.max(sgram[:, :np.minimum(10, scols)], axis=1), self.f_sd
        )
        # Store sthresh at each column, for debug
        # thr = np.zeros((srows, scols))
        peaks = np.zeros((srows, scols))
        # optimization of mask update
        __sp_pts = len(sthresh)
        __sp_v = self.__sp_vals

        for col in range(scols):
            s_col = sgram[:, col]
            # Find local magnitude peaks that are above threshold
            sdmaxposs = np.nonzero(locmax(s_col) * (s_col > sthresh))[0]
            # Work down list of peaks in order of their absolute value
            # above threshold
            valspeaks = sorted(zip(s_col[sdmaxposs], sdmaxposs), reverse=True)
            for val, peakpos in valspeaks[:self.maxpksperframe]:
                # What we actually want
                # sthresh = spreadpeaks([(peakpos, s_col[peakpos])],
                #                      base=sthresh, width=f_sd)
                # Optimization - inline the core function within spreadpeaks
                sthresh = np.maximum(sthresh,
                                     val * __sp_v[(__sp_pts - peakpos):
                                                  (2 * __sp_pts - peakpos)])
                peaks[peakpos, col] = 1
            sthresh *= a_dec
        return peaks

    def _decaying_threshold_bwd_prune_peaks(self, sgram, peaks, a_dec):
        """ backwards pass of findpeaks """
        scols = np.shape(sgram)[1]
        # Backwards filter to prune peaks
        sthresh = self.spreadpeaksinvector(sgram[:, -1], self.f_sd)
        for col in range(scols, 0, -1):
            pkposs = np.nonzero(peaks[:, col - 1])[0]
            peakvals = sgram[pkposs, col - 1]
            for val, peakpos in sorted(zip(peakvals, pkposs), reverse=True):
                if val >= sthresh[peakpos]:
                    # Setup the threshold
                    sthresh = self.spreadpeaks([(peakpos, val)], base=sthresh,
                                               width=self.f_sd)
                    # Delete any following peak (threshold should, but be sure)
                    if col < scols:
                        peaks[peakpos, col] = 0
                else:
                    # delete the peak
                    peaks[peakpos, col - 1] = 0
            sthresh = a_dec * sthresh
        return peaks

    def find_peaks(self, d=None, sr=None, sgram=None, backward=False):
        """ Find the local peaks in the spectrogram as basis for fingerprints.
            Returns a list of (time_frame, freq_bin) pairs.

        :params:
          d - np.array of float
            Input waveform as 1D vector

          sr - int
            Sampling rate of d (not used)

        sgram - np.array of float
            Spectrogram of d, as returned by librosa.stft

        backward - bool
            If True, do a backward pass to prune peaks.  If False, don't.

        :returns:
          pklist - list of (int, int)
            Ordered list of landmark peaks found in STFT.  First value of
            each pair is the time index (in STFT frames, i.e., units of
            n_hop/sr secs), second is the FFT bin (in units of sr/n_fft
            Hz).
        
          peaks - np.array of int
            Constellation map of peaks in sgram.  peaks[i, j] is True if the
            spectrogram has a peak at time i and frequency j.
        """

        if d is None and sgram is None:
            raise ValueError("find_peaks: must specify d or sgram")

        if sgram is not None:
            # we've been given a spectrogram, so we don't need to calculate it
            pass
        else:
            if len(d) == 0:
                return []
            mywin = np.hanning(self.n_fft + 2)[1:-1]
            sgram = np.abs(librosa.stft(y=d, n_fft=self.n_fft,
                                    hop_length=self.n_hop,
                                    window=mywin))
        logmelspec = librosa.amplitude_to_db(sgram, ref=np.max)
        # masking envelope decay constant
        a_dec = (1 - 0.01 * (self.density * np.sqrt(self.n_hop / 352.8) / 35)) ** (1 / self.oversamp)

        sgrammax = np.max(sgram)
        if sgrammax > 0.0:
            sgram = np.log(np.maximum(sgram, np.max(sgram) / 1e6))
            sgram = sgram - np.mean(sgram)
        else:
            # The sgram is identically zero, i.e., the input signal was identically
            # zero.
            print("find_peaks: Warning: input signal is identically zero.")
            return None, None
        # High-pass filter onset emphasis
        sgram = np.array([scipy.signal.lfilter([1, -1],
                                               [1, -self.hpf_pole ** (1 / self.oversamp)], s_row)
                          for s_row in sgram])
        # Prune to keep only local maxima in spectrum that appear above an online,
        # decaying threshold
        peaks = self._decaying_threshold_fwd_prune(sgram, a_dec)
        # Further prune these peaks working backwards in time, to remove small peaks
        # that are closely followed by a large peak
        if backward:
            peaks = self._decaying_threshold_bwd_prune_peaks(sgram, peaks, a_dec)
        # build a list of peaks we ended up with
        scols = np.shape(sgram)[1]
        pklist = []
        for col in range(scols):
            for bin_ in np.nonzero(peaks[:, col])[0]:
                amp = logmelspec[bin_, col]
                pklist.append([col/sgram.shape[1], bin_/sgram.shape[0], amp])
        return peaks, np.array(pklist)


    def wavfile2peaks(self, filename):
        """ Read a soundfile and return its landmark peaks as a
            list of (time, bin) pairs.  If specified, resample to sr first.
            shifts > 1 causes hashes to be extracted from multiple shifts of
            waveform, to reduce frame effects. REMOVED FROM IMPLEMENTATION  """
        try:
            d, sr = librosa.load(filename, sr=self.target_sr)
        except Exception as e:  
            message = "wavfile2peaks: Error reading " + filename
            if self.fail_on_error:
                print(e)
                raise IOError(message)
            print(message, "skipping")
            d = []
            sr = self.target_sr

        peaks, pklist = self.find_peaks(d, sr)

        return peaks, pklist



class GPUPeakExtractorv2(nn.Module):
    def __init__(self, cfg):
        super(GPUPeakExtractorv2, self).__init__()

        self.blur_kernel = cfg['blur_kernel']
        self.n_filters = cfg['n_filters']
        self.stride =cfg['peak_stride']
        # self.convs = nn.Sequential(
        #     nn.Conv2d(3, self.n_filters, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(self.n_filters),
        #     nn.ReLU(),
        #     nn.Conv2d(self.n_filters, self.n_filters, 3, stride=2, padding=1),
        #     nn.BatchNorm2d(self.n_filters),
        #     nn.ReLU(),
        #     nn.Conv2d(self.n_filters, self.n_filters, 3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.n_filters),
        # )

        self.convs = nn.Sequential(
            nn.Conv2d(3, 
                      self.n_filters, 
                      kernel_size=self.blur_kernel, 
                      stride=(self.stride, 1), 
                      padding=(self.blur_kernel[0] // 2, self.blur_kernel[1] // 2)
                      ),
            nn.ReLU(),
        )

        # Get number of GPUs
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.n_gpus == 0:
            self.n_gpus = 1

        T_tensor = torch.linspace(0, 1, steps=cfg['n_frames'])
        T_tensor = T_tensor.unsqueeze(0).unsqueeze(1).repeat(cfg['bsz_train'] // self.n_gpus, cfg['n_mels'], 1)
        self.T_tensor = T_tensor

        F_tensor = torch.linspace(0, 1, steps=cfg['n_mels'])
        F_tensor = F_tensor.unsqueeze(0).unsqueeze(2).repeat(cfg['bsz_train'] // self.n_gpus, 1, cfg['n_frames'])
        self.F_tensor = F_tensor
        # Initialize conv layer with kaiming initialization
        self.init_weights()


    def peak_from_features(self, features, as_mask=False):
        # Find local maxima along the time axis
        maxima_time = F.max_pool2d(features, kernel_size=(1, 3), stride=1, padding=(0, 1))
        maxima_time = torch.eq(features, maxima_time)

        # Find local maxima along the frequency axis
        maxima_freq = F.max_pool2d(features, kernel_size=(3, 1), stride=1, padding=(1, 0))
        maxima_freq = torch.eq(features, maxima_freq)

        # Combine maxima along both axes to get a binary matrix
        peaks = (maxima_time & maxima_freq).float()     

        if as_mask:
            mask = peaks * features
            # Set all non-zero values to 1; this get rid of zero values that were picked up as peaks
            mask[mask != 0] = 1.0
            return mask
        
        else:
            return peaks * features
        

        # Initialize conv layer with kaiming initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, spec_tensor):
        # Normalize the spectrogram
        min_vals = torch.amin(spec_tensor, dim=(1, 2), keepdim=True)
        max_vals = torch.amax(spec_tensor, dim=(1, 2), keepdim=True)
        spec_tensor = (spec_tensor - min_vals) / (max_vals - min_vals)

        # peaks = self.peak_from_features(spec_tensor.unsqueeze(1))
        peaks = spec_tensor.unsqueeze(1)

        # Put postional tensors to the same device as the peaks tensor
        self.T_tensor = self.T_tensor.to(peaks.device)
        self.F_tensor = self.F_tensor.to(peaks.device)

        # Concatenate T_tensor, F_tensor and spec_tensor to get a tensor of shape (batch, 3, H, W)
        try:
            tensor = torch.cat((self.T_tensor.unsqueeze(1), self.F_tensor.unsqueeze(1), peaks), dim=1)
        except RuntimeError as e:
            # Validation case
            T_tensor = torch.linspace(0, 1, steps=spec_tensor.shape[2], device=spec_tensor.device)
            T_tensor = T_tensor.unsqueeze(0).unsqueeze(1).repeat(spec_tensor.shape[0], spec_tensor.shape[1], 1)
            F_tensor = torch.linspace(0, 1, steps=spec_tensor.shape[1], device=spec_tensor.device)
            F_tensor = F_tensor.unsqueeze(0).unsqueeze(2).repeat(spec_tensor.shape[0], 1, spec_tensor.shape[2])
            tensor = torch.cat((T_tensor.unsqueeze(1), F_tensor.unsqueeze(1), peaks), dim=1)

        feature = self.convs(tensor)
        # print(f"Log: Convolution completed with shape {feature.shape}")
        # self.l1 = torch.norm(feature, p=1)

        B, C, H, W = feature.shape
        feature = feature.reshape(B, C, -1)
        return feature