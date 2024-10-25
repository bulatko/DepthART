import rocket

from mde.capsules.meter.metrics_fn import *
from mde.capsules.meter.ibims_metrics import dbe_acc_metric, dbe_com_metric, pe_fla_metric, pe_ori_metric


class DepthMetric(rocket.Metric):
    def __init__(
            self,
            metric_fn,
            tag: str = "metric",
            required_keys: list[tuple[str, str]] = [("target", "out"), ("predict", "pout"), ("mask", "mask")],
            allowed_datasets: list[str] | None = None,
            normalize_by_image: bool = True
    ):
        super().__init__()
        self.enum = 0.0
        self.denum = 0.0
        self.metric_fn = metric_fn
        self.tag = tag
        self.required_keys = required_keys
        self.allowed_datasets = set(allowed_datasets) if allowed_datasets is not None else None
        self.normalize_by_image = normalize_by_image

    def value(self):
        if self.denum == 0:
            return 0
        return float(self.enum / self.denum)

    def launch(self, attrs=None):
        if attrs is None or attrs.batch is None:
            return

        if (self.allowed_datasets is not None) and attrs.batch.metadata.name[0] not in self.allowed_datasets:
            return

        for batch_idx in range(len(attrs.batch.target)):
            cur_enum, cur_denum = self.metric_fn(**{
                arg_name: attrs.batch[batch_name][batch_idx]
                for batch_name, arg_name in self.required_keys
            })
            if cur_denum > 0:
                if self.normalize_by_image:
                    self.enum += cur_enum / cur_denum
                    self.denum += 1
                else:
                    self.enum += cur_enum
                    self.denum += cur_denum

        if attrs.looper is not None:
            attrs.looper.state.update(
                {self.tag: self.value()}
            )

    def reset(self, attrs=None):
        if attrs is None or attrs.batch is None:
            return
        if attrs.looper is not None:

            name = f"{attrs.looper.tag}:{self.tag}"
            attrs.looper.state.update({name: self.value()})
        else:
            name = self.tag

        if attrs.tracker is not None:
            attrs.tracker.scalars.update({name: self.value()})

        self.enum, self.denum = 0.0, 0.0


class MAE(DepthMetric):
    def __init__(self):
        super().__init__(mae, tag="mae")


class MSE(DepthMetric):
    def __init__(self):
        super().__init__(mse, tag="mse")


class RMSE(DepthMetric):
    def __init__(self):
        super().__init__(rmse, tag="rmse")


class AbsRel(DepthMetric):
    def __init__(self):
        super().__init__(absrel, tag="abs-rel")


class SQRel(DepthMetric):
    def __init__(self):
        super().__init__(sqrel, tag="sq-rel")


class D1(DepthMetric):
    def __init__(self):
        super().__init__(d1, tag="d-1.25")


class D2(DepthMetric):
    def __init__(self):
        super().__init__(d2, tag="d-1.25**2")


class D3(DepthMetric):
    def __init__(self):
        super().__init__(d3, tag="d-1.25**3")


class D102(DepthMetric):
    def __init__(self):
        super().__init__(d102, tag="d-1.02")


class D105(DepthMetric):
    def __init__(self):
        super().__init__(d105, tag="d-1.05")


class D110(DepthMetric):
    def __init__(self):
        super().__init__(d1, tag="d-1.10")


class DBEAcc(DepthMetric):
    def __init__(self):
        super().__init__(
            dbe_acc_metric, tag='dbe-acc',
            allowed_datasets=['IBIMS1'],
            required_keys=[("edges", "edges"), ("predict", "pred")]
        )


class DBECom(DepthMetric):
    def __init__(self):
        super().__init__(
            dbe_com_metric, tag='dbe-com',
            allowed_datasets=['IBIMS1'],
            required_keys=[("edges", "edges"), ("predict", "pred")]
        )


class PlanarityBaseMetric(DepthMetric):
    def __init__(self, fn, tag):
        super().__init__(
            fn, tag=tag,
            allowed_datasets=['IBIMS1'],
            normalize_by_image=False,  # It is per-plane, not per-image metric
            required_keys=[
                ("predict", "predict"),
                ("mask", "mask"),
                ("target", "target"),
                ("mask_table", "mask_table"),
                ("mask_floor", "mask_floor"),
                ("mask_walls", "mask_walls"),
                ("table_planes", "table_planes"),
                ("floor_planes", "floor_planes"),
                ("wall_planes", "wall_planes"),
                ("camera", "camera")
            ]
        )


class PlanarityFlatnessError(PlanarityBaseMetric):
    def __init__(self):
        super().__init__(pe_fla_metric, tag='pe-fla')


class PlanarityOrientationError(PlanarityBaseMetric):
    def __init__(self):
        super().__init__(pe_ori_metric, tag='pe-ori')
