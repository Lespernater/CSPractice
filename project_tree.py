# don't forget your imports!

###NOTE: The indentation of the instructions is intentional -- it may help with determining where code should go in
####     terms ofconditions.

class ProjectTree:
    """
    A class for building a tree containing words.
    """

    def __init__(self):
        """
        A constructor for the ProjectTree.
        """
        ##########################YOUR CODE HERE##########################
        # should have a root node for an instance attribute

    def get_root(self):
        """
        A method for returning the root of the ProjectTree.
        :return: the root of the tree
        """
        ##########################YOUR CODE HERE##########################

    def insert(self, word):
        """
        A method for inserting words into a ProjectTree.
        :param word: the word to be inserted
        :return: None
        """
        ##########################YOUR CODE HERE##########################
        # 1. Start at the root.

        # 2. Iterate through each letter in the word.

            # 3. If the letter isn't a child of the current node, add it to the children of the current node.

            # 4. Make the current node the child node containing the next letter in the word.

        # 5. Mark the final node as the end of a word.

    def find(self, word):
        """
        A method for finding a word in the ProjectTree.
        :param word: the word to be found
        :return: True, if found;
                 False, if not found.
        """
        ##########################YOUR CODE HERE##########################
        # 1. Start at the root.

        # 2. Iterate through each letter in the word.

            # 3. If the letter isn't a child of the current node, return False.

            # Otherwise, make the appropriate child node the current node.

        # 4. If the last letter is a word, return True

        # Otherwise, return False

    def delete(self, node, word, index):
        """
        A method for deleting words from a ProjectTree using recursion.
        :param node: the node to start at (the first call should use the root)
        :param word: the word being deleted from the tree
        :param index: an index to track the character of the node being considered for deletion
        :return: True, if it's safe to delete the node;
                 False, if it isn't safe.
        """
        ##########################YOUR CODE HERE##########################
        # 1. Check if the word is in the ProjectTree. If it isn't, we return False.

        # 2. Store the index of the character in the word.

        # 3. Set the current node to be the node associated with that character.

        # 4. Create a boolean value to track whether or not the node can safely be deleted.

        # 5. If the letter is the final letter of the word

            # And if the current node has any children

                # Change the node so that it is no longer marked as the end of a word and return False
                # handles the condition where the word being removed is the prefix of other words in the tree

            # if it has no children, it is safe to delete it,

                # so remove it from the children and return True

        # 6. If the current node has more than one child, then it isn't safe to delete.
        #    Recursively call delete using the index of the next character in the word and return False.
        # this handles the condition in which a prefix is shared with another word

        # 7. If the node is the end of a word

            # recursively call delete using the index of the next character in the word and return False.

        # 8. Recursively call delete on the next character in the word and store the result in the boolean value from Step 4.

        # 9. If it's safe to delete (that boolean is helpful here)

            # then remove that node and return True

        # Otherwise, return False

    def traversal(self, node, words, word=''):
        """
        A method for traversing the tree and returning all of its words.
        :param node: the node we are starting from
        :param words: the list of words
        :param word: the word to be added to the list
        :return: a list of words contained in the ProjectTree
        """
        ##########################YOUR CODE HERE##########################
        # 1. base case - if the node is a word,

            # we add it to the words list (if not already present)

        # 2. recursive case - traverse through the children of the node

            # make sure to add the letters so you get the full word

        # 3. Return the list of words.

    def word_starts_with(self, prefix):
        """
        A method for returning all of the words that start with a given prefix.
        :param prefix: the prefix for the words
        :return: a list of words beginning with that prefix
        """
        ##########################YOUR CODE HERE##########################
        # 1. Start with the root.

        # 2. Create an empty list for storing words.

            # if the letter isn't stored as a child

                # return an empty list

            # 3. Make the child node the new node.

        # 4. Traverse the tree, using the current node, the list for storing words, and the prefix being searched for as
        #    parameters.

        # 5. Return the words list.

    def build_project_tree(self, filename):
        """
        A method for building a ProjectTree from a file.
        :param filename: the name of the file containing words to be inserted
        :return: None
        """
        ##########################YOUR CODE HERE##########################
        # 1. Open the file and insert each entry into your tree.




