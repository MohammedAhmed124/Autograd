
class GlobalConfig:
    backward_mode = False

    def is_propagating_backwards():
        return GlobalConfig.backward_mode