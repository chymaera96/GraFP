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



def peaks2mask(peaks, patch_shape=(8, 6)):
    """ Divide the spectrogram into patch regions. If the patch contains a peak,
        set the mask to 1. Otherwise, set the mask to 0."""
    
    # Divide mask in to 8x8 regions
    h, w = peaks.shape
    nrows = patch_shape[0]
    ncols = patch_shape[1]

    assert h % nrows == 0, f"Height {h} is not divisible by nrows {nrows}"
    assert w % ncols == 0, f"Width {w} is not divisible by ncols {ncols} "

    mask = peaks.reshape(h // nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols)

    # Set mask to 1 if patch contains a peak
    for ix in range(mask.shape[0]):
        if mask[ix].sum() > 0:
            mask[ix] = 1
        else:
            mask[ix] = 0

    # Reshape mask to original shape
    mask = mask.reshape(h // nrows, -1, nrows, ncols).swapaxes(1, 2).reshape(h, w)
    return mask


class GPUPeakExtractor(nn.Module):
    def __init__(self, cfg):
        super(GPUPeakExtractor, self).__init__()
        self.pad_length = cfg['n_peaks']
        self.blur_kernel = cfg['blur_kernel']
        self.blur_sigma = cfg['blur_sigma']


    def peak_from_features(self, features):
         # Find local maxima along the time axis
        maxima_time = F.max_pool2d(features.unsqueeze(0), kernel_size=(1, 3), stride=1, padding=(0, 1))
        maxima_time = torch.eq(features, maxima_time.squeeze(0))

        # Find local maxima along the frequency axis
        maxima_freq = F.max_pool2d(features.unsqueeze(1), kernel_size=(3, 1), stride=1, padding=(1, 0))
        maxima_freq = torch.eq(features, maxima_freq.squeeze(1))

        # Combine maxima along both axes to get a binary matrix
        peaks = (maxima_time & maxima_freq).float()  

        # Normalize the spectrogram
        min_vals = torch.amin(features, dim=(1, 2), keepdim=True)
        max_vals = torch.amax(features, dim=(1, 2), keepdim=True)
        features = (features - min_vals) / (max_vals - min_vals)     

        return peaks * features

    def forward(self, spec_tensor):

        peaks = self.peak_from_features(spec_tensor)
        feature = gaussian_blur(peaks, kernel_size=self.blur_kernel, sigma=self.blur_sigma)
        peaks = self.peak_from_features(feature)

        # Compute nonzero indices once for the entire batch
        nonzero_indices = torch.nonzero(peaks)

        batch_nonzero_points = []
        for ix in range(spec_tensor.shape[0]):

            # Select indices for this item
            item_indices = nonzero_indices[nonzero_indices[:, 0] == ix][:, 1:]

            if item_indices.size(0) > 0:
                # Get the corresponding values
                nonzero_values = peaks[ix][item_indices[:, 0], item_indices[:, 1]]

                # Combine indices and values
                nonzero_points = torch.cat((item_indices.float(), nonzero_values.unsqueeze(1)), dim=1)

                # Normalize indices by dividing by the maximum dimension value
                nonzero_points[:, :2] /= torch.tensor([spec_tensor.shape[1], spec_tensor.shape[2]], device=spec_tensor.device)

                # # Setup for 2D peaks
                # nonzero_points = item_indices.float()
                # # Normalize indices by dividing by the maximum dimension value
                # nonzero_points /= torch.tensor([spec_tensor.shape[1], spec_tensor.shape[2]], device=spec_tensor.device)

                # Pad points or truncate to a fixed size
                pad_length = self.pad_length - nonzero_points.size(0)
                if pad_length < 0:
                    print(f"Warning: truncating points; there are {nonzero_points.size(0)} points")
                    # print(f"Maximum possible peaks is {spec_tensor.shape[1]} * {spec_tensor.shape[2]}")
                    # print(f"nonzero_values shape {nonzero_values.shape}")
                    nonzero_points = nonzero_points[:self.pad_length].transpose(1,0)
                    batch_nonzero_points.append(nonzero_points)
                else:
                    padded_points = F.pad(nonzero_points, (0, 0, 0, pad_length), mode='constant', value=0).transpose(1,0)
                    batch_nonzero_points.append(padded_points)

        return torch.stack(batch_nonzero_points)



class GPUPeakExtractorv2(nn.Module):
    def __init__(self, cfg):
        super(GPUPeakExtractorv2, self).__init__()

        self.blur_kernel = cfg['blur_kernel']
        self.n_filters = cfg['n_filters']
        self.stride =cfg['peak_stride']
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.n_filters, kernel_size=self.blur_kernel, stride=(self.stride, 1), padding=self.blur_kernel//2),
            nn.ReLU(),
        )

        # Initialize conv layer with kaiming initialization
        self.init_weights()

    def peak_from_features(self, features, mask=False):
        # Find local maxima along the time axis
        maxima_time = F.max_pool2d(features, kernel_size=(1, 3), stride=1, padding=(0, 1))
        maxima_time = torch.eq(features, maxima_time)

        # Find local maxima along the frequency axis
        maxima_freq = F.max_pool2d(features, kernel_size=(3, 1), stride=1, padding=(1, 0))
        maxima_freq = torch.eq(features, maxima_freq)

        # Combine maxima along both axes to get a binary matrix
        peaks = (maxima_time & maxima_freq).float()  

        # # Normalize the spectrogram
        # min_vals = torch.amin(features, dim=(1, 2), keepdim=True)
        # max_vals = torch.amax(features, dim=(1, 2), keepdim=True)
        # features = (features - min_vals) / (max_vals - min_vals)     

        if mask:
            return peaks
        
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

        # assert spec_tensor.device == torch.device('cuda:0'), f"Input tensor must be on GPU. Instead found on {spec_tensor.device}"

        peaks = self.peak_from_features(spec_tensor.unsqueeze(1))
        feature = self.conv(peaks)
        self.l1 = torch.norm(feature, p=2)
        peaks = self.peak_from_features(feature)

        T_tensor = torch.arange(feature.shape[3], device=feature.device) / feature.shape[3]
        T_tensor = T_tensor.unsqueeze(0).unsqueeze(0).repeat(feature.shape[0], 
                                                             self.n_filters, 
                                                             feature.shape[2], 1)
                
        F_tensor = torch.arange(feature.shape[2], device=spec_tensor.device) / spec_tensor.shape[2]
        F_tensor = F_tensor.unsqueeze(0).transpose(0,1).unsqueeze(0).repeat(feature.shape[0],
                                                                            self.n_filters, 1,
                                                                            feature.shape[3])
        
        # Concatenate T_tensor, F_tensor and feature to get a tensor of shape (batch, 3, C, H, W)
        tensor = torch.cat((T_tensor.unsqueeze(1), F_tensor.unsqueeze(1), feature.unsqueeze(1)), dim=1)

        # Repeat peaks to match the shape of tensor
        peaks = peaks.unsqueeze(1).repeat(1, tensor.shape[1], 1, 1, 1)
        tensor = tensor * peaks

        B, _, C, F, T = tensor.shape
        tensor = tensor.reshape(B, 3, C, -1)
    
        return tensor