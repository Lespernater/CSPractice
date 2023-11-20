class ProjectNode:
    """
    A class of nodes meant for a ProjectTree.
    """

    def __init__(self, letter=''):
        """
        A constructor for a ProjectNode
        :param letter: the node's key;
                        an empty string, by default
        """
        ##########################YOUR CODE HERE##########################
        # Needs instance variables for the key, identifying if the node is the end of a word,
        # and to hold a dictionary of child nodes

    def get_letter(self):
        """
        Returns the node's key.
        :return: the node's key, a character/letter
        """
        ##########################YOUR CODE HERE##########################

    def get_children(self):
        """
        Returns the node's children.
        :return: the node's children, a dictionary
        """
        ##########################YOUR CODE HERE##########################

    def add_child(self, character, node):
        """
        Adds a child to the children dictionary.
        :param character: the key, a letter
        :param node: the value, representing the node
        :return: None
        """
        ##########################YOUR CODE HERE##########################

    def end_word(self, truth_value):
        """
        Sets the node to be the end of the word.
        :param truth_value: a boolean - True if end of word; False otherwise
        :return: None
        """
        ##########################YOUR CODE HERE##########################

    def is_word(self):
        """
        A getter that tells us whether the node's key is the end of a word.
        :return: True if it is the end of a word;
                 False otherwise
        """
        ##########################YOUR CODE HERE##########################

    def __str__(self):
        """
        A string representation of the node.
        :return: a string containing the key
        """
        ##########################YOUR CODE HERE##########################




