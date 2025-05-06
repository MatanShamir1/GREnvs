
def get_property_reference(obj: object, property_name: str):
	"""
	Returns a reference to a property in an object given its name.

	Args:
		obj: The object containing the property
		property_name: The name of the property to find

	Returns:
		A reference to the property if found, None otherwise
	"""

	# Handle the case where the property is directly in the object
	if hasattr(obj, property_name):
		return obj

	# If the property isn't found directly, search recursively
	for attr_name in dir(obj):
		# Skip special attributes
		if attr_name.startswith('__'):
			continue

		try:
			attr_value = getattr(obj, attr_name)

			# Skip None and primitive types
			if attr_value is None or isinstance(attr_value, (int, float, str, bool)):
				continue

			# Recursively search in nested objects
			if hasattr(attr_value, '__dict__'):
				result = get_property_reference(attr_value, property_name)
				if result is not None:
					return result

		except Exception:
			# Skip any attributes that can't be accessed
			continue

	return None

