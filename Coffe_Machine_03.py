# The CoffeeMachine class is responsible for initialization of states, managing state transitions and to get current state. 
class CoffeeMachine:
    def __init__(self):
        # Initialize all states and set the initial state to 'a'.
        self.state_a = StateA('a', self)  # Creating an instance of StateA and pass the coffee machine instance to it.
        self.state_b = StateB('b', self)  # Creating an instance of StateB.
        self.state_c = StateC('c', self)  # Creating an instance of StateC.
        self.state_d = StateD('d', self)  # Creating an instance of StateD.
        self.state_d_prime = StateDPrime('d_prime', self)  # Creating an instance of StateDPrime.
        self.state_e = StateE('e', self)  # Creating an instance of StateE.
        self.state_f = StateF('f', self)  # Creating an instance of StateF.

        self.current_state = self.state_a  # Setting the initial state of the coffee machine to state 'a'.

    
    def transition(self, input_alphabet):
        return self.current_state.transition(input_alphabet)  
    
    # Method to retrieve the current state's name.
    def get_current_state(self):
        return self.current_state.name  


# Base class representing a state in the coffee machine's FSM
class State:
    def __init__(self, name, coffee_machine):
        self.name = name  
        self.coffee_machine = coffee_machine 

    # Abstract method to handle input, which will be implemented by subclasses.
    def handle_input(self, input_alphabet):
        raise NotImplementedError("This method should be implemented by subclasses")

    # Abstract method to handle output, which will be implemented by subclasses.
    def handle_output(self, input_alphabet):
        raise NotImplementedError("This method should be implemented by subclasses")

    # Abstract method to transition to the next state. To be implemented by subclasses.
    def transition(self, input_alphabet):
        raise NotImplementedError("This method should be implemented by subclasses")


# StateA class representing the coffee machine being in state 'a'.
class StateA(State):
    def handle_input(self, input_alphabet):
        # Handle the input and return the appropriate output.
        if input_alphabet == 'clean':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'pod':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'water':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'button':
            return self.handle_output(input_alphabet)  
        return 'error'  

    def handle_output(self, input_alphabet):
        # Define the output based on the input.
        if input_alphabet in ['clean', 'pod', 'water']:
            return 'OK'  
        elif input_alphabet == 'button':
            return 'error'  
        return 'error' 

    # Transition method encapsulates the logic to move to the next state.
    def transition(self, input_alphabet):
        output_alphabet = self.handle_input(input_alphabet)
        if input_alphabet == 'clean':
            self.coffee_machine.current_state = self 
        elif input_alphabet == 'pod':
            self.coffee_machine.current_state = self.coffee_machine.state_b  
        elif input_alphabet == 'water':
            self.coffee_machine.current_state = self.coffee_machine.state_c  
        elif input_alphabet == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_f  
        return output_alphabet  


# StateB class representing the coffee machine being in state 'b'.
class StateB(State):
    def handle_input(self, input_alphabet):
        if input_alphabet == 'clean':
            return self.handle_output(input_alphabet) 
        elif input_alphabet == 'pod':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'water':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'button':
            return self.handle_output(input_alphabet)  
        return 'error'  

    def handle_output(self, input_alphabet):
        if input_alphabet in ['clean', 'pod', 'water']:
            return 'OK' 
        elif input_alphabet == 'button':
            return 'error'  #
        return 'error'  

    def transition(self, input_alphabet):
        output_alphabet = self.handle_input(input_alphabet)  
        if input_alphabet == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a 
        elif input_alphabet == 'pod':
            self.coffee_machine.current_state = self 
        elif input_alphabet == 'water':
            self.coffee_machine.current_state = self.coffee_machine.state_d  
        elif input_alphabet == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_f  
        return output_alphabet  


# StateC class representing the coffee machine being in state 'c'.
class StateC(State):
    def handle_input(self, input_alphabet):
        if input_alphabet == 'clean':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'water':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'pod':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'button':
            return self.handle_output(input_alphabet)  
        return 'error'  

    def handle_output(self, input_alphabet):
        if input_alphabet in ['clean', 'pod', 'water']:
            return 'OK' 
        elif input_alphabet == 'button':
            return 'error'  
        return 'error'  

    def transition(self, input_alphabet):
        output_alphabet = self.handle_input(input_alphabet)  
        if input_alphabet == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a 
        elif input_alphabet == 'water':
            self.coffee_machine.current_state = self  
        elif input_alphabet == 'pod':
            self.coffee_machine.current_state = self.coffee_machine.state_d_prime  
        elif input_alphabet == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_f  
        return output_alphabet  


# StateD class representing the coffee machine being in state 'd'.
class StateD(State):
    def handle_input(self, input_alphabet):
        if input_alphabet in ['water', 'pod']:
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'button':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'clean':
            return self.handle_output(input_alphabet)  
        return 'error'  

    def handle_output(self, input_alphabet):
        if input_alphabet in ['water', 'pod']:
            return 'OK' 
        elif input_alphabet == 'button':
            return 'coffee produced'  
        return 'error' 

    def transition(self, input_alphabet):
        output_alphabet = self.handle_input(input_alphabet)  
        if input_alphabet in ['water', 'pod']:
            self.coffee_machine.current_state = self  
        elif input_alphabet == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_e  
        elif input_alphabet == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a  
        return output_alphabet  


# StateDPrime class representing the coffee machine being in state 'd_prime'.
class StateDPrime(State):
    def handle_input(self, input_alphabet):
        if input_alphabet in ['water', 'pod']:
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'button':
            return self.handle_output(input_alphabet)  
        elif input_alphabet == 'clean':
            return self.handle_output(input_alphabet)  
        return 'error'  

    def handle_output(self, input_alphabet):
        if input_alphabet in ['water', 'pod']:
            return 'OK'  
        elif input_alphabet == 'button':
            return 'coffee produced'  
        return 'error'  

    def transition(self, input_alphabet):
        output_alphabet = self.handle_input(input_alphabet)  
        if input_alphabet in ['water', 'pod']:
            self.coffee_machine.current_state = self  
        elif input_alphabet == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_e  
        elif input_alphabet == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a 
        return output_alphabet  

# StateE class representing the coffee machine being in state 'e' 
class StateE(State):
    def handle_input(self, input_alphabet):
        if input_alphabet == 'clean':
            return self.handle_output(input_alphabet)  
        return 'error' 

    def handle_output(self, input_alphabet):
        if input_alphabet == 'clean':
            return 'OK'  
        return 'error'  

    def transition(self, input_alphabet):
        output_alphabet = self.handle_input(input_alphabet)  
        if input_alphabet == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a  
        else:
            self.coffee_machine.current_state = self.coffee_machine.state_f  
        return output_alphabet  


# StateF class representing the coffee machine being in state 'f' (error state).
class StateF(State):
    def handle_input(self, input_alphabet):
        return self.handle_output(input_alphabet)  
    def handle_output(self, input_alphabet):
        return 'error'  
    def transition(self, input_alphabet):
        return self.handle_input(input_alphabet)  


# Interactive simulation function to test the behaviour of the coffee machine.
def interactive_coffee_machine():
    coffee_machine = CoffeeMachine()  # Instantiate the coffee machine.

    print("Welcome to the Coffee Machine Simulator!")
    
    # Infinite loop for continuous interaction with the coffee machine.
    while True:
        current_state = coffee_machine.get_current_state()  # Get the current state.
        print(f"\nCurrent state: {current_state}")
        
        # Get user input for the next action (input alphabet).
        user_input = input("Enter an input alphabet (water, pod, button, clean) or 'exit' to stop: ").lower()

        if user_input == 'exit':  # Exit condition to stop the simulation.
            print("Exiting the coffee machine simulation. Goodbye!")
            break
        
        # Validate the input to ensure it's one of the allowed options.
        if user_input not in ['water', 'pod', 'button', 'clean']:
            print("Invalid input. Please enter 'water', 'pod', 'button', or 'clean'.")
            continue

        # Perform state transition based on the user input.
        output_alphabet = coffee_machine.transition(user_input)
        
        # Display the result after the state transition.
        new_state = coffee_machine.get_current_state()
        print(f"Input: {user_input} -> Output: {output_alphabet}, New State: {new_state}")



if __name__ == '__main__':
    interactive_coffee_machine()
