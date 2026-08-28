from selfupdatingai.ai_interface import AIInterface

def test_ai_interface():
    ai_interface = AIInterface()
    expected_response1 = "The sky looks blue because molecules in the atmosphere scatter blue light more than other colors due to its shorter wavelength, making blue light more visible during the day."
    test_message1 = "In one sentence, explain why the sky looks blue."
    response1 = ai_interface.send_message(test_message1,deterministic=True)
    print(response1)

    assert(response1==expected_response1)

    test_message2 = "What's the dominant wavelength range. Answer with a hyphenated range only."
    response2 = ai_interface.send_message(test_message2,deterministic=True)
    expected_response2 = "450-495 nm"
    print(response2)

    assert(response2==expected_response2)

if __name__=="__main__":
    test_ai_interface()