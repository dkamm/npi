class Enum:
    @classmethod
    def get_name(cls, enum):
        for k, v in cls.__dict__.items():
            if v == enum:
                return k
        raise ValueError("invalid enum {}".format(enum))
