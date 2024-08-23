# The CoffeeMachine class is responsible for managing state transitions and interactions.
class CoffeeMachine:
    def __init__(self):
        # Initialize all states and set the initial state to 'a'.
        self.state_a = StateA('a', self)
        self.state_b = StateB('b', self)
        self.state_c = StateC('c', self)
        self.state_d = StateD('d', self)
        self.state_d_prime = StateDPrime('d_prime', self)
        self.state_e = StateE('e', self)
        self.state_f = StateF('f', self)

        self.current_state = self.state_a  # Starting state is 'a'.

    # This method simply delegates the transition to the current state's transition method.
    def transition(self, input_signal):
        return self.current_state.transition(input_signal)

    # Method to retrieve the current state's name.
    def get_current_state(self):
        return self.current_state.name


# Base class representing a state in the coffee machine's finite state machine (FSM).
class State:
    def __init__(self, name, coffee_machine):
        self.name = name  # The name of the state, such as 'a', 'b', etc.
        self.coffee_machine = coffee_machine  # Reference to the coffee machine instance.

    # Abstract method to handle input. To be implemented by subclasses.
    def handle_input(self, input_signal):
        raise NotImplementedError("This method should be implemented by subclasses")

    # Abstract method to handle output. To be implemented by subclasses.
    def handle_output(self, input_signal):
        raise NotImplementedError("This method should be implemented by subclasses")

    # Abstract method to transition to the next state. To be implemented by subclasses.
    def transition(self, input_signal):
        raise NotImplementedError("This method should be implemented by subclasses")


# StateA class representing the coffee machine being in state 'a'.
class StateA(State):
    def handle_input(self, input_signal):
        if input_signal == 'clean':
            return self.handle_output(input_signal)
        elif input_signal == 'pod':
            return self.handle_output(input_signal)
        elif input_signal == 'water':
            return self.handle_output(input_signal)
        elif input_signal == 'button':
            return self.handle_output(input_signal)
        return 'error'

    def handle_output(self, input_signal):
        if input_signal in ['clean', 'pod', 'water']:
            return 'OK'
        elif input_signal == 'button':
            return 'error'
        return 'error'

    # Transition method encapsulates the logic to move to the next state.
    def transition(self, input_signal):
        output = self.handle_input(input_signal)
        if input_signal == 'clean':
            self.coffee_machine.current_state = self  # Stays in state 'a'.
        elif input_signal == 'pod':
            self.coffee_machine.current_state = self.coffee_machine.state_b  # Move to state 'b'.
        elif input_signal == 'water':
            self.coffee_machine.current_state = self.coffee_machine.state_c  # Move to state 'c'.
        elif input_signal == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_f  # Move to state 'f'.
        return output


# StateB class representing the coffee machine being in state 'b'.
class StateB(State):
    def handle_input(self, input_signal):
        if input_signal == 'clean':
            return self.handle_output(input_signal)
        elif input_signal == 'pod':
            return self.handle_output(input_signal)
        elif input_signal == 'water':
            return self.handle_output(input_signal)
        elif input_signal == 'button':
            return self.handle_output(input_signal)
        return 'error'

    def handle_output(self, input_signal):
        if input_signal in ['clean', 'pod', 'water']:
            return 'OK'
        elif input_signal == 'button':
            return 'error'
        return 'error'

    def transition(self, input_signal):
        output = self.handle_input(input_signal)
        if input_signal == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a  # Move to state 'a'.
        elif input_signal == 'pod':
            self.coffee_machine.current_state = self  # Stays in state 'b'.
        elif input_signal == 'water':
            self.coffee_machine.current_state = self.coffee_machine.state_d  # Move to state 'd'.
        elif input_signal == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_f  # Move to state 'f'.
        return output


# StateC class representing the coffee machine being in state 'c'.
class StateC(State):
    def handle_input(self, input_signal):
        if input_signal == 'clean':
            return self.handle_output(input_signal)
        elif input_signal == 'water':
            return self.handle_output(input_signal)
        elif input_signal == 'pod':
            return self.handle_output(input_signal)
        elif input_signal == 'button':
            return self.handle_output(input_signal)
        return 'error'

    def handle_output(self, input_signal):
        if input_signal in ['clean', 'pod', 'water']:
            return 'OK'
        elif input_signal == 'button':
            return 'error'
        return 'error'

    def transition(self, input_signal):
        output = self.handle_input(input_signal)
        if input_signal == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a  # Move to state 'a'.
        elif input_signal == 'water':
            self.coffee_machine.current_state = self  # Stays in state 'c'.
        elif input_signal == 'pod':
            self.coffee_machine.current_state = self.coffee_machine.state_d_prime  # Move to state 'd_prime'.
        elif input_signal == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_f  # Move to state 'f'.
        return output


# StateD class representing the coffee machine being in state 'd'.
class StateD(State):
    def handle_input(self, input_signal):
        if input_signal in ['water', 'pod']:
            return self.handle_output(input_signal)
        elif input_signal == 'button':
            return self.handle_output(input_signal)
        elif input_signal == 'clean':
            return self.handle_output(input_signal)
        return 'error'

    def handle_output(self, input_signal):
        if input_signal in ['water', 'pod']:
            return 'OK'
        elif input_signal == 'button':
            return 'coffee produced'
        return 'error'

    def transition(self, input_signal):
        output = self.handle_input(input_signal)
        if input_signal in ['water', 'pod']:
            self.coffee_machine.current_state = self  # Stays in state 'd'.
        elif input_signal == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_e  # Move to state 'e'.
        elif input_signal == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a  # Move to state 'a'.
        return output


# StateDPrime class representing the coffee machine being in state 'd_prime'.
class StateDPrime(State):
    def handle_input(self, input_signal):
        if input_signal in ['water', 'pod']:
            return self.handle_output(input_signal)
        elif input_signal == 'button':
            return self.handle_output(input_signal)
        elif input_signal == 'clean':
            return self.handle_output(input_signal)
        return 'error'

    def handle_output(self, input_signal):
        if input_signal in ['water', 'pod']:
            return 'OK'
        elif input_signal == 'button':
            return 'coffee produced'
        return 'error'

    def transition(self, input_signal):
        output = self.handle_input(input_signal)
        if input_signal in ['water', 'pod']:
            self.coffee_machine.current_state = self  # Stays in state 'd_prime'.
        elif input_signal == 'button':
            self.coffee_machine.current_state = self.coffee_machine.state_e  # Move to state 'e'.
        elif input_signal == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a  # Move to state 'a'.
        return output


# StateE class representing the coffee machine being in state 'e' (coffee produced state).
class StateE(State):
    def handle_input(self, input_signal):
        if input_signal == 'clean':
            return self.handle_output(input_signal)
        return 'error'

    def handle_output(self, input_signal):
        if input_signal == 'clean':
            return 'OK'
        return 'error'

    def transition(self, input_signal):
        output = self.handle_input(input_signal)
        if input_signal == 'clean':
            self.coffee_machine.current_state = self.coffee_machine.state_a  # Move to state 'a'.
        else:
            self.coffee_machine.current_state = self.coffee_machine.state_f  # Move to state 'f'.
        return output


# StateF class representing the coffee machine being in state 'f' (error state).
class StateF(State):
    def handle_input(self, input_signal):
        return self.handle_output(input_signal)

    def handle_output(self, input_signal):
        return 'error'  # Always returns 'error' for any input signal.

    def transition(self, input_signal):
        return self.handle_input(input_signal)  # Always stays in state 'f'.


# Interactive simulation function for the coffee machine.
def interactive_coffee_machine():
    coffee_machine = CoffeeMachine()  # Instantiate the coffee machine.

    print("Welcome to the Coffee Machine Simulator!")
    
    # Infinite loop for continuous interaction with the coffee machine.
    while True:
        current_state = coffee_machine.get_current_state()  # Get the current state.
        print(f"\nCurrent state: {current_state}")
        
        # Get user input for the next action (input signal).
        user_input = input("Enter an input signal (water, pod, button, clean) or 'exit' to stop: ").lower()

        if user_input == 'exit':  # Exit condition to stop the simulation.
            print("Exiting the coffee machine simulation. Goodbye!")
            break
        
        # Validate the input to ensure it's one of the allowed options.
        if user_input not in ['water', 'pod', 'button', 'clean']:
            print("Invalid input. Please enter 'water', 'pod', 'button', or 'clean'.")
            continue

        # Perform state transition based on the user input.
        output = coffee_machine.transition(user_input)
        
        # Display the result after the state transition.
        new_state = coffee_machine.get_current_state()
        print(f"Input: {user_input} -> Output: {output}, New State: {new_state}")


# Entry point of the program. Starts the interactive coffee machine simulation.
if __name__ == '__main__':
    interactive_coffee_machine()
