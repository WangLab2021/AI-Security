import torch
from argparse import Namespace


def DI(y_hat, y, s1):
    prediction = y_hat
    y = (y > 0.0).int()

    z_0_mask = (s1 == 0.0)
    z_1_mask = (s1 == 1.0)
    z_0 = int(torch.sum(z_0_mask))
    z_1 = int(torch.sum(z_1_mask))

    Pr_y_hat_1_z_0 = float(torch.sum((prediction == 1)[z_0_mask])) / z_0 if z_0 > 0 else 0.
    Pr_y_hat_1_z_1 = float(torch.sum((prediction == 1)[z_1_mask])) / z_1 if z_1 > 0 else 0.

    Pr_y_hat_0_z_0 = float(torch.sum((prediction == 0)[z_0_mask])) / z_0 if z_0 > 0 else 0.
    Pr_y_hat_0_z_1 = float(torch.sum((prediction == 0)[z_1_mask])) / z_1 if z_1 > 0 else 0.

    y_1_z_0_mask = (y == 1.0) & (s1 == 0.0)
    y_1_z_1_mask = (y == 1.0) & (s1 == 1.0)
    y_0_z_0_mask = (y == 0.0) & (s1 == 0.0)
    y_0_z_1_mask = (y == 0.0) & (s1 == 1.0)

    y_1_z_0 = int(torch.sum(y_1_z_0_mask))
    y_1_z_1 = int(torch.sum(y_1_z_1_mask))
    y_0_z_0 = int(torch.sum(y_0_z_0_mask))
    y_0_z_1 = int(torch.sum(y_0_z_1_mask))

    Pr_y_hat_1_y_1_z_0 = float(torch.sum((prediction == 1)[y_1_z_0_mask])) / y_1_z_0 if y_1_z_0 > 0 else 0
    Pr_y_hat_1_y_1_z_1 = float(torch.sum((prediction == 1)[y_1_z_1_mask])) / y_1_z_1 if y_1_z_1 > 0 else 0


    # FPR = FP/(FP+TN)
    # FPR_z_0本来是0，给了1
    y_hat_0_y_1_z_0 = float(torch.sum((prediction == 0)[y_1_z_0_mask]))
    FNR_z_0 = float(y_hat_0_y_1_z_0) / y_1_z_0 if y_1_z_0 > 0 else 0.
    y_hat_0_y_1_z_1 = float(torch.sum((prediction == 0)[y_1_z_1_mask]))
    FNR_z_1 = float(y_hat_0_y_1_z_1) / y_1_z_1 if y_1_z_1 > 0 else 0.

    TPR_z_0 = 1.0 - FNR_z_0  # y_hat_1_y_1_z_0
    TPR_z_1 = 1.0 - FNR_z_1  # y_hat_1_y_1_z_1

    y_hat_1_y_0_z_0 = float(torch.sum((prediction == 1)[y_0_z_0_mask]))
    FPR_z_0 = float(y_hat_1_y_0_z_0) / y_0_z_0 if y_0_z_0 > 0 else 0.
    y_hat_1_y_0_z_1 = float(torch.sum((prediction == 1)[y_0_z_1_mask]))
    FPR_z_1 = float(y_hat_1_y_0_z_1) / y_0_z_1 if y_0_z_1 > 0 else 0.

    TNR_z_0 = 1.0 - FPR_z_0  # y_hat_0_y_0_z_0
    TNR_z_1 = 1.0 - FPR_z_1  # y_hat_0_y_0_z_1


    # delta EO =
    # y = 0   |y_hat_1_y_0_z_0 - y_hat_1_y_0_z_1|
    # y = 1   |y_hat_1_y_1_z_0 - y_hat_1_y_1_z_1|
    EOP = abs(FPR_z_0 - FPR_z_1) + abs(TPR_z_0 - TPR_z_1)

    min_dp = min(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)
    max_dp = max(Pr_y_hat_1_z_0, Pr_y_hat_1_z_1)
    if max_dp == 0:
        dis_imp = 0.0
    else:
        dis_imp = min_dp / max_dp

    return dis_imp, Pr_y_hat_1_z_0, Pr_y_hat_1_z_1, FNR_z_0, FNR_z_1, FPR_z_0, FPR_z_1, EOP


di_data = Namespace(y_hat=None, y=None, s1=None)
output_old = model(inputs)              # fc之后的结果 num_classes
if di_data.y_hat is None:
    di_data.y_hat = (output_old.data.max(1)[1]).int().reshape(-1, )
    di_data.y = copy.deepcopy(targets)
    di_data.s1 = copy.deepcopy(s1)
else:
    temp = (output_old.data.max(1)[1]).int().reshape(-1, )
    di_data.y_hat = torch.cat([di_data.y_hat, temp])
    di_data.y = torch.cat([di_data.y, targets])                 # target = class label
    di_data.s1 = torch.cat([di_data.s1, s1])                    # s1 =  sensitive attributes
dis_imp, Pr_y_hat_1_z_0, Pr_y_hat_1_z_1, FNR_z_0, FNR_z_1, FPR_z_0, FPR_z_1, EOP = DI(di_data.y_hat, di_data.y, di_data.s1)
DEO = abs(Pr_y_hat_1_z_0 - Pr_y_hat_1_z_1)