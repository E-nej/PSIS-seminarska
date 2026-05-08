class CommunicationModule:

    def __init__(self, neighbors):
        self.neighbors = neighbors
        self.messages = {}

    def update_message(self, tl_id, message):
        self.messages[tl_id] = message

    def get_neighbor_messages(self, tl_id):
        output = []

        for neighbor_id in self.neighbors.get(tl_id, []):
            if neighbor_id in self.messages:
                output.extend(self.messages[neighbor_id])
            else:
                output.extend([0, 0, 0, 0])

        return output