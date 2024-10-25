import torch



class Normalizer(torch.nn.Module):
    def __init__(self, affine=False, alpha=1.0, beta=0.0, training=False):
        super().__init__()
        self.affine = affine
        self.training = training

        if self.affine:
            assert alpha > 0.0, "Alpha cannot be zero"

            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
            self.register_buffer("beta", torch.tensor(beta, dtype=torch.float))

    def normalize(self, target, mask):
        raise NotImplementedError()

    def encode(self, target, mask):
        if not self.affine:
            return target

        target[mask] = self.alpha * target[mask] + self.beta

        return target

    def decode(self, data, mask=None):
        assert data.ndim == 4, "Normalizer expects 4 dim data tensor"

        if not self.affine:
            return data

        if mask is None:
            mask = torch.ones_like(data).bool()

        data[mask] = (data[mask] - self.beta) / self.alpha
        return data

    def forward(self, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target).bool()

        assert target.ndim == 4, "Normalizer expects 4 dim target tensor"
        assert mask.ndim == 4, "Normalizer expects 4 dim mask tensor"

        targets = torch.split(target, 1)
        masks = torch.split(mask, 1)

        normalizer_targets = list()
        for target, mask in zip(targets, masks):
            normalizer_targets.append(
                self.encode(self.normalize(target, mask), mask)
            )
           
        normalizer_target = torch.cat(normalizer_targets, dim=0)


        return normalizer_target


class NormalizerUTSSQuant(Normalizer):
    def __init__(self, quant=0.98, affine=False, alpha=1.0, beta=0.0, eps=1e-7):
        super().__init__(affine, alpha, beta)
        self.quant = quant
        self.eps = eps

    def normalize(self, target, mask):
        m_target = target[mask]

        shift = torch.quantile(m_target, 1.0 - self.quant)
        scale = torch.quantile(m_target, self.quant) - shift
        
        if self.training:
            target = (target - shift) / (scale + self.eps)
        else:
            target[mask] = (target[mask] - shift) / (scale + self.eps)
        return target


class NormalizerUTSSL1(Normalizer):
    def __init__(self, quant=0.98, affine=False, alpha=1.0, beta=0.0, eps=1e-7):
        super().__init__(affine, alpha, beta)
        self.quant = quant
        self.eps = eps

    def normalize(self, target, mask):
        m_target = target[mask]

        shift = m_target.median()
        scale = (m_target - shift).abs().mean()
        
        if self.training:
            target = (target - shift) / (scale + self.eps)
        else:
            target[mask] = (target[mask] - shift) / (scale + self.eps)
        return target


class NormalizerUTSL1(Normalizer):
    def __init__(self, quant=0.98, affine=False, alpha=1.0, beta=0.0, eps=1e-7):
        super().__init__(affine, alpha, beta)
        self.quant = quant
        self.eps = eps

    def normalize(self, target, mask):
        m_target = target[mask]
        
        if self.training:
            target = target / target.abs().mean()
        else:
            target[mask] = target[mask] / target[mask].abs().mean()
        return target


class NormalizerUTSQuant(Normalizer):
    def __init__(self, quant=0.98, affine=False, alpha=1.0, beta=0.0, eps=1e-7):
        super().__init__(affine, alpha, beta)
        self.quant = quant
        self.eps = eps

    def normalize(self, target, mask):
        m_target = target[mask]

        scale = torch.quantile(m_target, self.quant)

        if self.training:
            target = target / (scale + self.eps)
        else:
            target[mask] = m_target / (scale + self.eps)
        return target


class NormalizerUTSSGaussian(Normalizer):
    def __init__(self, affine=False, alpha=1.0, beta=0.0, eps=1e-7):
        super().__init__(affine, alpha, beta)
        self.eps = eps

    def normalize(self, target, mask):
        m_target = target[mask]

        shift, scale = torch.mean(m_target), torch.std(m_target)
        target[mask] = (m_target - shift) / (scale + self.eps)
        return target


class MarigoldNormalizer(NormalizerUTSSQuant):
    def __init__(self):
        super().__init__(affine=True,
                         alpha=2.0,
                         beta=0.5)
