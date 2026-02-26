# Copyright© 2025-2026 Gesellschaft zur Förderung der angewandten Forschung e.V.
# acting on behalf of its Fraunhofer Institut für Graphische Datenverarbeitung.
# Licensed under the EUPL. See LICENSE.txt.

from .backend import ArrayLike, shape
from .trainbase import TrainBase
from .utils import trains_match, block_tensor, sequence_product, get_device_dtype, namespace_of_trains

class ExactAddition:

    def __call__[T: ArrayLike](self,
                 *trains: TrainBase[T]
                 ) -> TrainBase[T]:
        trains_match(*trains)
        xp = namespace_of_trains(*trains)
        device, dtype = get_device_dtype(trains)

        if len(trains[0].shape) == 1:
            data = trains[0].data[0]
            for train in trains[1:]:
                data += train.data[0]
            return TrainBase(trains[0].shape, [data], copy_data=False)


        data = []
        for idx in range(len(trains[0].shape)):
            is_begin = (idx == 0)
            is_end = (idx == len(trains[0].shape)-1)

            tns = [train.data[idx] for train in trains]
            shapes = [shape(tn) for tn in tns]

            left = 1 if is_begin else sum(sh[0] for sh in shapes)
            middle = shapes[0][1:-1]
            right = 1 if is_end else sum(sh[-1] for sh in shapes)
            tmp = xp.zeros((left, *middle, right),
                           device=device, dtype=dtype)

            left_cut = 0 if is_begin else slice(None)
            right_cut = 0 if is_end else slice(None)
            for idxs in sequence_product(middle):
                cut = (left_cut, *idxs, right_cut)
                tmp[cut] = block_tensor(*[tn[cut] for tn in tns])
            data.append(tmp)

        return TrainBase(trains[0].shape, data, copy_data=False)
