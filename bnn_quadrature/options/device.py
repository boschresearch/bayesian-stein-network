from bnn_quadrature.options.enums import DeviceEnum


class MyDevice:
    def __init__(self):
        self._device = DeviceEnum.cpu

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val: DeviceEnum):
        self._device = val


my_device = MyDevice()
