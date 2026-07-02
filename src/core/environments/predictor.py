import torch
import torch.nn as nn
import numpy as np
from core.digital_twin.architectures import MLP
import copy
from collections import deque
import h5py as h5


class Predictor:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # create crank-angle degree (CAD) vectors
        # Full cycle (-360..360) at 0.1 CAD resolution -> 7200 samples.
        self.cad_plt = np.arange(-360, 360, 0.1)
        # For IMEP/work computations we use the center window (-180..180) -> 3600 samples,
        # which matches the predictor network output size used elsewhere.
        self.cad_window = self.cad_plt[:]
        self.res = 720 / self.cad_plt.size      # resolution of pressure trace (CAD/sample)
        self.cad = np.reshape(self.cad_plt, [1, -1, 1])

        # engine parameters for IMEP calc.
        b = 79 / 1000  # bore
        s = 86 / 1000  # stroke
        CR = 17.19  # compression ratio
        l = 160 / 1000  # connecting rod length
        a = s / 2  # crank radius
        delta = 0.6 / 1000  # piston pin offset
        R = l / a
        self.Vs = np.pi * ((b / 2) ** 2) * (2 * a)  # stroke volume
        Vc = self.Vs / (CR - 1)  # clearance volume
        self.V = Vc * (1 + (0.5 * (CR - 1)) * (R + 1 - np.cos(np.radians(self.cad_plt)) - np.sqrt(
            R ** 2 - (np.sin(np.radians(self.cad_plt)) + delta) ** 2)))  # Total Volume vector
        self.V = self.V[:]

        self._imep_history = deque(maxlen=20)
        self._imep_history.extend([0] * 20)

        # miscellaneous
        self.ones = np.ones([1, 7200, 3])

    def reset_transient_state(self) -> None:
        """Reset cycle-history features used for moving-average IMEP outputs."""
        self._imep_history.clear()
        self._imep_history.extend([0] * 20)
       
    
    def _extract_normalization_values(self, sample_data_path: str):
        with h5.File(sample_data_path, "r") as fin:
            mean = fin["normalization/feature_mean"][...].reshape(-1)
            std = fin["normalization/feature_std"][...].reshape(-1)
        return mean, std

    def init_model(self,
                   input_size: int,
                   num_layers: int,
                   layer_exp: int,
                   out_size: int,
                   dropout: float,
                   weights_path: str,
                   sample_data_path: str):
        # create model and load weights/checkpoint
        self.model = MLP(
            input_dim=input_size,
            output_dim=out_size,
            num_hidden=num_layers,
            hidden_exp=layer_exp,
            dropout=dropout,
        )
        self.model = self.model.to(self.device)
        checkpoint = torch.load(weights_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.mean, self.std = self._extract_normalization_values(sample_data_path)


    def format_data(self, data):
        data = (data - self.mean) / self.std
        data = np.float32(data)
        data = torch.tensor(data, device=self.device)
        return data

    def model_predict(self, values, *, noise_in_percent=None):

        # get data in correct format
        data = self.format_data(values)
        # make prediction and format it
        p = self.model(data)
        p = np.squeeze(p.detach().cpu().numpy())

        if p.ndim == 1:
            p = p.reshape(1, -1)
        elif p.ndim > 2:
            raise ValueError("p must be a 1D or 2D array.")

        work = self._calculate_work(p, self.V)
        imep = work / self.Vs

        mprr = self._calculate_mprr(p, int((-20 + 360)/self.res), int((40 + 360)/self.res))

        start_ind = int((-10 + 360)/0.1)
        mfb = self._calculate_mfb(p, self.V, self.cad_plt, start_ind)
        SOC, CA10, CA50, CA90, EOC = self._calculate_combustion_timings(mfb, start_ind)
        CA10_90 = CA90 - CA10
        hrr, qnet = self._calculate_q_net(p, self.V, SOC, EOC)
        Pmax = np.max(p, axis=1)
        ivc_ind = 2000
        Pint = p[:, ivc_ind]
        self._imep_history.append(float(imep))
        IMEP_ma, skewness = self._calculate_moving_average_and_skewness(self._imep_history)
        # add noise if desired
        if noise_in_percent is not None:
            imep = imep + np.random.normal(0, np.abs(imep) * noise_in_percent / 100)
            mprr = mprr + np.random.normal(0, np.abs(mprr) * noise_in_percent / 100)
        # Return a CAD vector aligned with `p`
        cad = self.cad_window if p.shape[0] == self.V.shape[0] else self.cad_plt

        # create output dictionary
        output = {
            "imep": imep,
            "mprr": mprr,
            "CA50": CA50,
            "CA10_CA90": CA10_90,
            "Net_heat_release": qnet,
            "Pint": Pint,
            "Pressure_max": Pmax,
            "IMEP_moving_average": IMEP_ma,
            "Skewness_moving_average": skewness
        }

        return p, cad, output


    def _calculate_work(self, P: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Function to calculate the work done by the engine."""
        return np.trapz(P, x=V, axis=1)

    def _calculate_mprr(self, P: np.ndarray, ind_start: int, ind_end: int) -> np.ndarray:
        """Function to calculate the maximum pressure rise rate."""
        PRR = (P[:, 2:] - P[:, :-2]) / (2 * self.res)
        return np.max(PRR[:,ind_start:ind_end],axis=1)

    def _calculate_q_net(self, P: np.ndarray, V: np.ndarray, SOC: np.ndarray, EOC: np.ndarray) -> np.ndarray:
        """Function to calculate the heat release rate and net heat release."""
        P = copy.deepcopy(P)*100000    # Convert to Pa
        dV = np.pad((V[2:] - V[:-2])/(2*0.1), pad_width=(1,1), mode='constant', constant_values=(0,0))
        dP = np.pad((P[:,2:] - P[:,:-2])/(2*0.1), pad_width=((0,0), (1,1)), mode='constant', constant_values=(0,0))
        gamma = 1.34    # This is a guess but should be fine if all cycles are the same
        indices = np.tile(np.arange(P.shape[1]), (P.shape[0],1))
        mask = (indices >= SOC[:, None]) & (indices <= EOC[:, None])
        hrr =  (gamma * P * dV)/(gamma-1) + (V * dP)/(gamma-1)
        qnet = np.sum(np.where(mask, hrr, 0),axis=1)*0.1
        return hrr, qnet

    def _calculate_mfb(self, P: np.ndarray, V: np.ndarray, cad: np.ndarray, start_ind: int) -> np.ndarray:
        """Function to calculate the mfb."""
        end_ind = int((20 + 360)/0.1)
        V_ref = np.min(V)
        gamma = 1.34    # This is a guess but should be fine if all cycles are the same
        cad_mid = 0.5 *(cad[:-1] + cad[1:])
        P_motored = P[:,:-1] * (V[:-1] / V[1:])**gamma
        dP_comb = P[:,1:] - P_motored
        dP_comb_corrected = dP_comb * (V[1:] / V_ref)**gamma
        dP_window = dP_comb_corrected[:,start_ind:end_ind]
        cumulative = np.cumsum(dP_window,axis=1)
        total = cumulative[:,-1]
        xb = cumulative / total.reshape(-1,1)
        return xb

    def _calculate_combustion_timings(self, mfb: np.ndarray, start_ind: int) -> np.ndarray:
        """Function to calculate SOC, CA10, CA50, CA90, EOC."""
        soc_mask = mfb > 0.005
        SOC = soc_mask.argmax(axis=1) + start_ind
        ca10_mask = mfb > 0.1
        CA10 = ca10_mask.argmax(axis=1) + start_ind
        ca50_mask = mfb > 0.5
        CA50 = ca50_mask.argmax(axis=1) + start_ind
        ca90_mask = mfb > 0.9
        CA90 = ca90_mask.argmax(axis=1) + start_ind
        eoc_mask = mfb == 1
        EOC = eoc_mask.argmax(axis=1) + start_ind
        outs = np.array([SOC, CA10, CA50, CA90, EOC])
        return outs

    def _calculate_moving_average_and_skewness(self, x: deque):
        # convert to numpy array
        arr = np.asarray(x)

        # Moving Average
        mean = np.mean(arr)

        # Skewness
        q1 = np.percentile(arr, 25)
        q2 = np.percentile(arr, 50)
        q3 = np.percentile(arr, 75)
        iqr = q3 - q1
        if iqr == 0:
            skew = -1.0
        else:
            skew = (q1 - 2 * q2 + q3) / iqr
        return mean, skew
