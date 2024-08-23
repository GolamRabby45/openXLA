"""
The `State` class represents a state in the coffee machine's state machine. Each state has a name and a reference to the coffee machine instance.

The `handle_input` method is an abstract method that must be implemented by subclasses to handle different input signals and transition to the next state accordingly.
"""
class State:
    def __init__(self, name, coffee_machine):
        self.name = name
        self.coffee_machine = coffee_machine

    def handle_input(self, input_signal):
        raise NotImplementedError("This method should be implemented by subclasses")


class StateA(State):
"""
    Represents the 'StateA' state in the coffee machine's state machine. This state handles various input signals and transitions to the appropriate next state.
    
    The `handle_input` method is responsible for processing the input signal and returning the appropriate response and next state. The possible input signals are:
    
    - 'clean': Transitions to the current state ('OK', self)
    - 'pod': Transitions to the 'StateB' state ('OK', self.coffee_machine.state_b)
    - 'water': Transitions to the 'StateC' state ('OK', self.coffee_machine.state_c)
    - 'button': Transitions to the 'StateF' state with an 'error' response ('error', self.coffee_machine.state_f)
    - Any other input signal: Returns an 'error' response and remains in the current state ('error', self)
    """
        def handle_input(self, input_signal):
        if input_signal == 'clean':
            return 'OK', self
        elif input_signal == 'pod':
            return 'OK', self.coffee_machine.state_b
        elif input_signal == 'water':
            return 'OK', self.coffee_machine.state_c
        elif input_signal == 'button':
            return 'error', self.coffee_machine.state_f
        return 'error', self


class StateB(State):
    def handle_input(self, input_signal):
        if input_signal == 'clean':
            return 'OK', self.coffee_machine.state_a
        elif input_signal == 'pod':
            return 'OK', self
        elif input_signal == 'water':
            return 'OK', self.coffee_machine.state_d
        elif input_signal == 'button':
            return 'error', self.coffee_machine.state_f
        return 'error', self


class StateC(State):
    def handle_input(self, input_signal):
        if input_signal == 'clean':
            return 'OK', self.coffee_machine.state_a
        elif input_signal == 'water':
            return 'OK', self
        elif input_signal == 'pod':
            return 'OK', self.coffee_machine.state_d_prime
        elif input_signal == 'button':
            return 'error', self.coffee_machine.state_f
        return 'error', self


class StateD(State):
    def handle_input(self, input_signal):
        if input_signal in ['water', 'pod']:
            return 'OK', self
        elif input_signal == 'button':
            return 'coffee produced', self.coffee_machine.state_e
        elif input_signal == 'clean':
            return 'OK', self.coffee_machine.state_a
        return 'error', self


class StateDPrime(State):
    def handle_input(self, input_signal):
        if input_signal in ['water', 'pod']:
            return 'OK', self
        elif input_signal == 'button':
            return 'coffee produced', self.coffee_machine.state_e
        elif input_signal == 'clean':
            return 'OK', self.coffee_machine.state_a
        return 'error', self


class StateE(State):
    def handle_input(self, input_signal):
        if input_signal == 'clean':
            return 'OK', self.coffee_machine.state_a
        else:
            return 'error', self.coffee_machine.state_f


class StateF(State):
    def handle_input(self, input_signal):
        return 'error', self


class CoffeeMachine:
    def __init__(self):
        # Initialize all states
        self.state_a = StateA('a', self)
        self.state_b = StateB('b', self)
        self.state_c = StateC('c', self)
        self.state_d = StateD('d', self)
        self.state_d_prime = StateDPrime('d_prime', self)
        self.state_e = StateE('e', self)
        self.state_f = StateF('f', self)

        # Start in state 'a'
        self.current_state = self.state_a

    def transition(self, input_signal):
        output, next_state = self.current_state.handle_input(input_signal)
        self.current_state = next_state
        return output

    def get_current_state(self):
        return self.current_state.name


def interactive_coffee_machine():
    coffee_machine = CoffeeMachine()

    print("Welcome to the Coffee Machine Simulator!")
    
    while True:
        current_state = coffee_machine.get_current_state()
        print(f"\nCurrent state: {current_state}")
        
        # Get user input
        user_input = input("Enter an input signal (water, pod, button, clean) or 'exit' to stop: ").lower()

        if user_input == 'exit':
            print("Exiting the coffee machine simulation. Goodbye!")
            break
        
        if user_input not in ['water', 'pod', 'button', 'clean']:
            print("Invalid input. Please enter 'water', 'pod', 'button', or 'clean'.")
            continue

        # Perform state transition
        output = coffee_machine.transition(user_input)
        
        # Display the result
        new_state = coffee_machine.get_current_state()
        print(f"Input: {user_input} -> Output: {output}, New State: {new_state}")


if __name__ == '__main__':
    interactive_coffee_machine()
