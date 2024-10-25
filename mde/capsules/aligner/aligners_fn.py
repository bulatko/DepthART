import torch



def align_ssi_l2(predict, target, mask):
    #
    #   Ax = b => x = (A.T A)^-1 A.T b
    #
    #                 | pred_d1, 1 |
    #   A = A[N, 2] = | pred_d2, 1 |
    #                 | ...        |
    #
    #   b = b[N] = [target_d1, ... target_dN].T
    #

    m_predict = predict[mask]
    m_target = target[mask]

    ones = torch.ones_like(m_predict)
    m_predict = torch.stack([m_predict, ones], dim=1)

    solution = torch.inverse(m_predict.T @ m_predict) @ m_predict.T @ m_target
    solution = solution.squeeze()
    
    return predict * solution[0] + solution[1]



def align_ssi_l1(predict, target, mask):
    
    m_predict = predict[mask]
    m_target = target[mask]
    
    t = m_target.median()
    s = (m_predict - t).abs().mean()

    solution = [s, t]

    return predict * solution[0] + solution[1]



def align_si_l2(predict, target, mask):
    #
    #   Ax = b => x = (A.T A)^-1 A.T b
    #
    #   A = A[N] = [pred_d1, ... pred_dN].T
    #
    #   b = b[N] = [target_d1, ... target_dN].T
    #

    m_predict = predict[mask]
    m_target = target[mask]

    solution = torch.reciprocal(m_predict.T @ m_predict) * (m_predict.T @ m_target)
    solution = solution.squeeze()

    return predict * solution


def align_si_l1(predict, target, mask):
    m_predict = predict[mask]
    m_target = target[mask]
    solution = (m_target / m_predict).median()
    return predict * solution


def align_si_l1log(predict, target, mask, log_domain=False):
    m_predict = predict[mask]
    m_target = target[mask]

    if not log_domain:
        return predict + (m_target - m_predict).median()
    else:
        solution = predict.log() + (m_target.log() - m_predict.log()).median()
        return solution.exp()
