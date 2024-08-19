import os
import stadiamaps
from stadiamaps.rest import ApiException
from pprint import pprint

# Placeholder for your Stadia Maps API key
api_key = '1844785d-cb2d-4478-bc79-442319986dc8'  # Replace with your actual API key

# Ensure the API key is correctly set
if not api_key or api_key == '123231122342':
    raise ValueError("You must replace the placeholder API key with your actual Stadia Maps API key.")

# Setting up the configuration
configuration = stadiamaps.Configuration()
configuration.api_key['ApiKeyAuth'] = api_key

# Optional: Setting the host if needed (e.g., to use the EU endpoint)
# configuration.host = "https://api-eu.stadiamaps.com"

# Enter a context with an instance of the API client
with stadiamaps.ApiClient(configuration) as api_client:
    # Create an instance of the Geocoding API class
    api_instance = stadiamaps.GeocodingApi(api_client)
    
    # The place name (address, venue name, etc.) to search for
    text = "Põhja pst 27a"  # Example address

    try:
        # Search and geocode based on partial input
        api_response = api_instance.autocomplete(text)
        print("The response of GeocodingApi->autocomplete:\n")
        pprint(api_response)
    except ApiException as e:
        print("Exception when calling GeocodingApi->autocomplete: %s\n" % e)

