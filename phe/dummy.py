def generate_paillier_keypair(n_length):
    return DummyKey(), DummyKey()


class DummyKey(object):
    def encrypt(self, message):
        return message

    def decrypt(self, message):
        return message
