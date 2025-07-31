import argparse
import struct
from pathlib import Path
from io import BytesIO
from typing import Dict, Any, Optional, Tuple
from wand.image import Image
from wand.exceptions import WandException


def get_image(path: str) -> bytes:
	"""
	Read image file as raw bytes.
	
	Args:
		path: File path to image
		
	Returns:
		Raw image data as bytes
		
	Raises:
		FileNotFoundError: If file cannot be read
	"""
	
	try:
		with open(path, "rb") as f:
			return f.read()
	except Exception as e:
		raise FileNotFoundError(e)


def convert_to(blob: bytes, format: str, new_format: str = "png") -> bytes:
	"""
	Convert image blob from one format to another using ImageMagick.
	
	Args:
		blob: Raw image data
		format: Source image format
		new_format: Target image format (default: png)
		
	Returns:
		Converted image data as bytes
		
	Raises:
		WandException: If ImageMagick conversion fails
		ValueError: If target format is unsupported
	"""

	format = format.lower()
	new_format = new_format.lower()

	try:
		with Image(blob=blob, format=format) as f:
			# In 2026-10 - end support of 3.10, use switch case instead
			# if format == "png":
			# elif format == "jpg":
			# elif format == "bmp":
			# elif format == "webp":
			# elif format == "tiff":
			# for now no special configs per format.
			f.format = new_format

			try:
				return f.make_blob()
			except WandException as e:
				raise ValueError(f"Unsupported format: \"{new_format}\":\n{e}")

	except WandException as e:
		raise WandException(f"Conversion to format: \"{format}\" failed:\n{e}")
	except Exception as e:
		raise Exception(f"Conversion to format: \"{format}\" failed:\n{e}")


def databend(data: bytes, method: str = "safe", pattern_type: str = "swap", pattern_from: bytes = b"\x99\x56", pattern_to: bytes = b"\x56\x99") -> bytes:
	"""
	Apply databending glitch effect by replacing specific byte patterns.
	Based on databending techniques from wikipedia.org/wiki/Databending and Signal Culture Cook Book by Michael Betancourt.
	
	Args:
		data: Raw image data to glitch.
		method: Databending method - "safe" preserves headers, "aggressive" affects entire file.
		pattern_type: Type of glitch - "swap", "increment", "bit_shift".
		
	Returns:
		Glitched image data with byte substitutions.
	"""
	
	if method == "safe":
		# Preserve headers
		header_size = min(1024, len(data) // 10)
		footer_size = min(512, len(data) // 20)
		
		if len(data) <= header_size + footer_size:
			return data # Too small to manipulate
		
		header = data[:header_size]
		middle = data[header_size:-footer_size]
		footer = data[-footer_size:]
		
		if pattern_type == "swap":
			glitched_middle = middle.replace(pattern_from, pattern_to)
		elif pattern_type == "increment":
			glitched_middle = middle.replace(pattern_from, pattern_to)
		elif pattern_type.startswith("bit_shift"):
			glitched_middle = _apply_bit_shift_patterns(middle, pattern_from, pattern_to, operation=pattern_type[10:])
		else:
			print(f"Unknown pattern type: \"{pattern_type}\", doing swap.")
			glitched_middle = data.replace(pattern_from, pattern_to)
		
		return header + glitched_middle + footer
	
	else: # Aggressive
		if pattern_type == "swap":
			return data.replace(pattern_from, pattern_to)
		elif pattern_type == "increment":
			return data.replace(pattern_from, pattern_to)
		elif pattern_type.startswith("bit_shift"):
			return _apply_bit_shift_patterns(data, pattern_from, pattern_to, operation=pattern_type[10:])
		else:
			print(f"Unknown pattern type: \"{pattern_type}\", doing swap.")
			return data.replace(pattern_from, pattern_to)


def _apply_bit_shift_patterns(data: bytes, pattern_from: bytes, pattern_to: bytes, operation: str = "xor") -> bytes:
	"""
	Apply bit shifting operations when pattern_from is found.
	
	Args:
		data: Input data to process.
		pattern_from: Pattern to search for.
		pattern_to: Pattern to use in bit operations.
		operation: Type of bit operation ("xor", "shift_right", "shift_left", "rotate_right", "rotate_left", "add").
	"""

	if len(pattern_from) != len(pattern_to):
		return data.replace(pattern_from, pattern_to)
	
	result = bytearray(data)
	pattern_len = len(pattern_from)
	
	i = 0
	while i <= len(result) - pattern_len:
		if result[i:i + pattern_len] == pattern_from:
			for j in range(pattern_len):
				original_byte = result[i + j]
				target_byte = pattern_to[j]
				
				if operation == "xor":
					result[i + j] = original_byte ^ target_byte
				elif operation == "shift_right":
					result[i + j] = original_byte >> 1
				elif operation == "shift_left":
					result[i + j] = (original_byte << 1) & 0xFF
				elif operation == "rotate_right":
					result[i + j] = ((original_byte >> 1) | (original_byte << 7)) # & 0xFF
				elif operation == "rotate_left":
					result[i + j] = ((original_byte << 1) | (original_byte >> 7)) & 0xFF
				elif operation == "add":
					result[i + j] = (original_byte + target_byte) & 0xFF
				else:
					# Default XOR
					result[i + j] = original_byte ^ target_byte
			
			i += pattern_len
		else:
			i += 1
	
	return bytes(result)


def validate_glitch_blob(blob: bytes, format: str) -> Dict[str, Any]:
	"""
	Validate and analyze potentially corrupted image blob for reconstruction viability.
	
	Args:
		blob: Raw image data (potentially corrupted).
		format: Expected image format.
		
	Returns:
		Dictionary containing validation results:
		- is_valid: bool indicating if blob passes ImageMagick validation.
		- format_detected: str of detected format based on signature.
		- size: int byte size of blob.
		- errors: list of validation errors.
		- warnings: list of non-fatal issues.
		- can_reconstruct: bool indicating reconstruction possibility.
		- metadata: dict of extracted image properties.
	"""

	validation = {
		"is_valid": False,
		"format_detected": None,
		"size": len(blob),
		"errors": [],
		"warnings": [],
		"can_reconstruct": False,
		"metadata": {}
	}
	
	try:
		# format signature validation
		format_signatures = {
			"png": [b"\x89PNG\r\n\x1a\n"],
			"jpeg": [b"\xff\xd8\xff"],
			"jpg": [b"\xff\xd8\xff"],
			"gif": [b"GIF87a", b"GIF89a"],
			"bmp": [b"BM"],
			"tiff": [b"II*\x00", b"MM\x00*"],
			"webp": [b"RIFF"]
		}
		
		format_lower = format.lower()
		if format_lower in format_signatures:
			signatures = format_signatures[format_lower]
			has_valid_signature = any(blob.startswith(sig) for sig in signatures)
			
			if has_valid_signature:
				validation["format_detected"] = format_lower
				validation["can_reconstruct"] = True
			else:
				validation["errors"].append(f"Invalid {format} signature")
		
		# Wand validation for additional metadata
		try:
			with Image(blob=blob) as img:
				validation["is_valid"] = True
				validation["metadata"] = {
					"width": img.width,
					"height": img.height,
					"format": img.format,
					"colorspace": str(img.colorspace),
					"depth": img.depth
				}
		except WandException as e:
			# primitive reconstruction might work
			validation["errors"].append(f"ImageMagick validation failed: {str(e)}")
			if validation["format_detected"]:
				validation["can_reconstruct"] = True
				validation["warnings"].append("Corrupted but may be reconstructable")
	
	except Exception as e:
		validation["errors"].append(f"Validation error: {str(e)}")
	
	return validation


def read_raw_pixels(blob: bytes, format: str, width: int = None, height: int = None) -> Tuple[bytes, Dict[str, Any]]:
	"""
	Extract raw pixel data from image blob using primitive parsing methods.
	
	Args:
		blob: Raw image data.
		format: Image format for parsing strategy.
		width: Expected image width (optional).
		height: Expected image height (optional).
		
	Returns:
		Tuple of (raw_pixel_data, metadata_dict).
	"""

	format_lower = format.lower()
	metadata = {"method": "primitive", "format": format_lower}
	
	if format_lower == "bmp":
		return _read_bmp_pixels_primitive(blob, metadata)
	elif format_lower in ["png"]:
		return _read_png_pixels_primitive(blob, metadata)
	elif format_lower in ["jpeg", "jpg"]:
		return _read_jpeg_pixels_primitive(blob, metadata)
	else:
		# desperate extraction; any RGB-like data patterns
		return _read_generic_pixels_primitive(blob, metadata, width, height)


def _read_bmp_pixels_primitive(blob: bytes, metadata: dict) -> Tuple[bytes, dict]:
	"""Parse BMP header and extract uncompressed pixel data."""

	try:
		if len(blob) < 54: # BMP header minimum size
			raise ValueError("BMP too small")
		
		# Read BMP header structure
		header = struct.unpack("<2sIHHIIIIHHIIIIII", blob[:54])
		width = header[6]
		height = header[7]
		bits_per_pixel = header[9]
		
		metadata.update({
			"width": width,
			"height": height,
			"bits_per_pixel": bits_per_pixel
		})
		
		pixel_offset = 54
		pixel_data = blob[pixel_offset:]
		
		return pixel_data, metadata
		
	except (struct.error, IndexError, ValueError) as e:
		metadata["error"] = f"BMP parsing failed: {e}"
		# Return raw data from reasonable offset as fallback
		return blob[54:] if len(blob) > 54 else blob, metadata


def _read_png_pixels_primitive(blob: bytes, metadata: dict) -> Tuple[bytes, dict]:
	"""Parse PNG chunks and extract IDAT compressed pixel data."""

	try:
		if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
			raise ValueError("Invalid PNG signature")
		
		pixel_data = b""
		offset = 8 # Skip PNG signature
		
		while offset < len(blob) - 8:
			try:
				chunk_length = struct.unpack(">I", blob[offset:offset+4])[0]
				chunk_type = blob[offset+4:offset+8]
				
				if chunk_type == b"IHDR" and chunk_length >= 13:
					# Get image dimensions from header
					ihdr_data = blob[offset+8:offset+8+13]
					width, height = struct.unpack(">II", ihdr_data[:8])
					metadata.update({"width": width, "height": height})
				elif chunk_type == b"IDAT":
					# Get compressed pixel data chunks
					idat_data = blob[offset+8:offset+8+chunk_length]
					pixel_data += idat_data
				
				offset += 4 + 4 + chunk_length + 4 # length + type + data + CRC
				
			except (struct.error, IndexError):
				break
		
		metadata["compressed"] = True
		return pixel_data, metadata
		
	except Exception as e:
		# Return PNG header then data
		metadata["error"] = f"PNG parsing failed: {e}"
		return blob[8:] if len(blob) > 8 else blob, metadata


def _read_jpeg_pixels_primitive(blob: bytes, metadata: dict) -> Tuple[bytes, dict]:
	"""Extract compressed JPEG scan data starting right after the SOS header."""

	try:
		if not blob.startswith(b"\xff\xd8"):
			raise ValueError("Invalid JPEG signature")
		
		# Find SOS marker (0xFFDA)
		sos_pos = blob.find(b"\xff\xda")
		if sos_pos == -1:
			raise ValueError("No SOS header marker found")
		
		# Parse SOS header 2-byte length field
		if sos_pos + 4 > len(blob):
			raise ValueError("Truncated SOS header")
		sos_length = struct.unpack(">H", blob[sos_pos + 2:sos_pos + 4])[0]
		
		# Validate length
		if sos_length < 2:
			raise ValueError("Invalid SOS header length")
		if sos_pos + sos_length > len(blob):
			raise ValueError("SOS header extends beyond blob")
		
		# Compressed pixel data starts immediately after SOS header
		scan_start = sos_pos + sos_length
		pixel_data = blob[scan_start:]
		
		metadata.update({
			"compressed": True,
			"sos_position": sos_pos,
			"sos_header_length": sos_length,
		})
		
		return pixel_data, metadata

	except Exception as e:
		metadata["error"] = f"JPEG parsing failed: {e}"
		# Fallback: return data after SOI (0xFFD8), if possible
		fallback = blob[2:] if len(blob) > 2 else blob
		return fallback, metadata


def _read_generic_pixels_primitive(blob: bytes, metadata: dict, width: int, height: int) -> Tuple[bytes, dict]:
	"""Generic pixel extraction for unknown or heavily corrupted formats."""

	metadata["method"] = "generic_primitive"
	
	if width and height:
		expected_rgb = width * height * 3
		expected_rgba = width * height * 4
		
		# Try different header offsets to find pixel-like data wuth common header sizes
		for offset in [0, 54, 138, 1078, 18, 26, 128, 512, 1024]:
			if offset < len(blob):
				remaining = len(blob) - offset
				if remaining >= expected_rgb * 0.7:
					metadata["assumed_format"] = "RGB"
					return blob[offset:offset + expected_rgb], metadata
				elif remaining >= expected_rgba * 0.7:
					metadata["assumed_format"] = "RGBA"
					return blob[offset:offset + expected_rgba], metadata
	
	# return largest chunk that may contain the pixel data... sorry
	return blob[min(1078, len(blob) // 4):], metadata


def reconstruct(blob: bytes, format: str, new_format: str = "png", output_filepath: str = None, primitive: bool = False, force_dimensions: Tuple[int, int] = None) -> bytes | None:
	"""
	Reconstruct image from potentially glitched blob with validation and primitive options.
	Can use either standard ImageMagick reconstruction or primitive pixel-level methods.
	Automatically falls back to primitive reconstruction if standard method fails.
	
	Args:
		blob: Raw image data.
		format: Source image format.
		new_format: Target format for output (default: png).
		output_filepath: Path to save file, returns None if provided.
		primitive: Use primitive pixel-level reconstruction instead of ImageMagick.
		force_dimensions: Force specific (width, height) for primitive mode.
		
	Returns:
		Reconstructed image bytes, or None if saved to file.
		
	Raises:
		WandException: If all reconstruction methods fail.
		ValueError: If blob cannot be reconstructed by any method.
	"""
	
	# Try reconstruction first unless primitive is forced
	if not primitive:
		try:
			validation = validate_glitch_blob(blob, format)
			if validation["can_reconstruct"]:
				return _standard_reconstruct(blob, format, new_format, output_filepath)
		except (WandException, Exception) as e:
			print(f"Standard reconstruction failed:\n{e}")
			print("Trying primitive reconstruction.")
	
	try:
		return _primitive_reconstruct(blob, format, new_format, output_filepath, force_dimensions)
	except Exception as e:
		raise WandException(f"All reconstruction methods failed:\n{e}")

def _standard_reconstruct(blob: bytes, format: str, new_format: str, output_filepath: str) -> bytes | None:
	"""Standard ImageMagick-based reconstruction."""

	try:
		with Image(blob=blob, format=format) as f:
			f.format = new_format
			if output_filepath:
				f.save(filename=output_filepath)
				return None
			else:
				return f.make_blob()
	except WandException as e:
		raise WandException(f"Standard reconstruction failed:\n{e}")


def _primitive_reconstruct(blob: bytes, format: str, new_format: str, output_filepath: str, force_dimensions: Tuple[int, int]) -> bytes | None:
	"""
	Primitive pixel-level reconstruction using raw data extraction.
	"""

	width, height = force_dimensions or (None, None)
	
	# Raw extraction
	pixel_data, metadata = read_raw_pixels(blob, format, width, height)
	
	if not width or not height:
		if "width" in metadata and "height" in metadata:
			width, height = metadata["width"], metadata["height"]
		else:
			# Guessing dimensions from data size
			if format.lower() == "png":
				estimated_pixels = len(blob) // 4 # bad estimate
				width = height = max(64, int(estimated_pixels ** 0.5))
			else:
				pixel_count = max(1, len(pixel_data) // 3) # RGB
				width = height = max(64, int(pixel_count ** 0.5))
	 
	width = max(64, min(width, 2048))
	height = max(64, min(height, 2048))
	
	try:
		# For corrupted PNG, extract data differently
		if format.lower() == "png" and len(pixel_data) < width * height * 3:
			result = _reconstruct_corrupted_png(blob, width, height, new_format, output_filepath)
			if result is not None or output_filepath is not None:
				return result
		
		# Create new image from raw pixel data
		with Image(width=width, height=height, background="black") as img:
			# Ensure there's enough data
			required_bytes = width * height * 3
			if len(pixel_data) < required_bytes:
				# Filling with corrupted data pattern (filler)
				pixel_data = (pixel_data * ((required_bytes // len(pixel_data)) + 1))[:required_bytes]
			
			try:
				img.import_pixels(0, 0, width, height, "RGB", "char", pixel_data[:required_bytes])
			except Exception as e:
				raise ValueError(f"Pixel import failed:\n{e}")
			img.format = new_format
			
			if output_filepath:
				img.save(filename=output_filepath)
				return None
			else:
				return img.make_blob()
				
	except Exception as e:
		# FAllback: create glitch art representation
		print(f"Primitive reconstruction warning: {e}")
		return _create_glitch_representation(pixel_data, width, height, new_format, output_filepath)


def _create_glitch_representation(data: bytes, width: int, height: int, format: str, output_filepath: str) -> bytes | None:
	"""Create visual representation of corrupted data when reconstruction fails."""

	# Ensure reasonable dimensions
	width = max(64, min(width or 256, 2048))
	height = max(64, min(height or 256, 2048))
	
	# Create RGB pixel array from corrupted data
	pixel_array = []
	data_len = len(data)
	
	for y in range(height):
		for x in range(width):
			idx = (y * width + x) % data_len # Wrap around if not enough data
			if idx < data_len:
				# Use data bytes to create glitch patterns
				color_val = data[idx] if isinstance(data[idx], int) else ord(data[idx])
				# Create more interesting glitch patterns
				r = color_val
				g = (color_val * 2) % 256
				b = (color_val + idx) % 256
				pixel_array.extend([r, g, b])
			else:
				pixel_array.extend([0, 0, 0]) # Black for missing data
	
	# Convert to bytes
	pixel_data = bytes(pixel_array)
	
	with Image(width=width, height=height, background="black") as img:
		# Import the glitch pixel data
		img.import_pixels(0, 0, width, height, "RGB", "char", pixel_data)
		img.format = format
		
		if output_filepath:
			img.save(filename=output_filepath)
			return None
		else:
			return img.make_blob()


def _reconstruct_corrupted_png(blob: bytes, width: int, height: int, format: str, output_filepath: str) -> bytes | None:
	"""
	Special handling for heavily corrupted PNG files.
	Makes glitch art by interpreting the corrupted data as visual patterns.
	"""

	start_offset = 8 # PNG signature
	data = blob[start_offset:]
	
	# Make pixel array from corrupted data
	pixel_array = []
	
	for y in range(height):
		for x in range(width):
			# Position in corrupted data
			pos = (y * width + x) % len(data)
			
			# Extract bytes and create color from corrupt patterns
			if pos < len(data) - 2:
				r = data[pos] if isinstance(data[pos], int) else ord(data[pos])
				g = data[pos + 1] if isinstance(data[pos + 1], int) else ord(data[pos + 1])
				b = data[pos + 2] if isinstance(data[pos + 2], int) else ord(data[pos + 2])
				
				# Apply glitch-like transformations, i.e fake glitch filler
				# Another workaround is to get that pixel from origin to fill
				r = (r ^ (pos & 0xFF)) % 256
				g = (g + (pos >> 8)) % 256
				b = (b * 2) % 256
				
				pixel_array.extend([r, g, b])
			else:
				pixel_array.extend([0, 0, 0])
	
	pixel_data = bytes(pixel_array)
	
	with Image(width=width, height=height, background="black") as img:
		# Import the glitch pixel data
		img.import_pixels(0, 0, width, height, "RGB", "char", pixel_data)		
		img.format = format
		
		if output_filepath:
			img.save(filename=output_filepath)
			return None
		else:
			return img.make_blob()


def _parse_bytes(data: str) -> bytes:
	try:
		return bytes.fromhex(data)
	except Exception as e:
		raise TypeError(f"Non valid bytes:\n{e}")


def _increment_bytes(data: bytes, by: int) -> bytes:
	byte_list = list(data)

	carry = by
	for i in range(len(byte_list) - 1, -1, -1):
		total = byte_list[i] + carry
		byte_list[i] = total % 256
		carry = total // 256
		if carry == 0:
			break
	# Affects size:
	# if carry:
	# 	byte_list = [carry] + byte_list
	
	return bytes(byte_list)


def _bytes_to_hex_string(byte_obj):
    return "".join(f"\\x{b:02X}" for b in byte_obj)


def main() -> None:
	parser = argparse.ArgumentParser(
		prog="AGAM",
		description="Makes supposedly appealing glitches in a wide varity of visuals.",
		epilog="Enjoy."
	)
	parser.add_argument("fileName", help="Original image file (e.g. input.png).")
	parser.add_argument("newFileName", help="Base name for output file (e.g. output.png).")
	parser.add_argument("--format", type=str, default="jpg", help="Format used to glitch (default: jpg).")
	parser.add_argument("--method", type=str, default="databend", help="Glitch method: databend, databend-aggressive (default: databend).")
	parser.add_argument("--pattern-type", type=str, default="swap", help="""Databending methods: swap, increment, bit_shift (default: swap).
Suffix bit_shift with: \"_xor\", \"_shift_right\", \"_shift_left\", \"_rotate_right\", \"_rotate_left\", \"_add\".""")
	parser.add_argument("--pattern-from", type=str, default="9956", help="Pattern to be replace from (default=9956).")
	parser.add_argument("--pattern-to", type=str, default="5699", help="Pattern to be replaced to (default=5699).")
	parser.add_argument("--iterations-from", action="store_true", help="Iterations affect pattern-from via incrementation.")
	parser.add_argument("--iterations-to", action="store_true", help="Iterations affect pattern-to via incrementation.")
	parser.add_argument("--primitive", action="store_true", help="Use primitive pixel-level reconstruction.")
	parser.add_argument("--iterations", type=int, default=1, help="Number of glitch iterations, applies to specific methods only (default: 1).")
	parser.add_argument("--dry-run", action="store_true", help="Waits for user confirmation between each iteration.")
	parser.add_argument("--verbose", action="store_true", help="Helps debugging.")
	args = parser.parse_args()
	
	# Workflow
	input_path = Path(args.fileName)
	input_name = input_path.stem
	input_format = input_path.suffix.lstrip(".")
	
	output_path = Path(args.newFileName)
	output_name = output_path.stem
	output_format = output_path.suffix.lstrip(".")

	output_folder: None
	if args.iterations > 1:
		output_folder = output_path.parent / f"{output_name}_glitches"
		output_folder.mkdir(exist_ok=True)
		
		if args.verbose:
			print(f"Created output folder: {output_folder}")

	image: bytes = get_image(args.fileName)
	converted_image: bytes = convert_to(blob=image, format=input_format, new_format=args.format)
	
	# Iterations handeling
	for i in range(args.iterations):
		glitch: bytes

		try:
			pattern_from = _parse_bytes(args.pattern_from)
			pattern_to = _parse_bytes(args.pattern_to)
		except Exception as e:
			print(f"Error parsing patterns:\n{e}")
			return

		if args.iterations_from:
			pattern_from = _increment_bytes(data=pattern_from, by=i)
		if args.iterations_to:
			pattern_to = _increment_bytes(data=pattern_to, by=i)
		
		if args.verbose:
			(input if args.dry_run else print)(f"Iteration {i+1}/{args.iterations} - {args.method}, {args.pattern_type}: {_bytes_to_hex_string(pattern_from)}  ->  {_bytes_to_hex_string(pattern_to)}")

		if args.method == "databend":
			glitch = databend(
				converted_image,
				method="safe",
				pattern_type=args.pattern_type,
				pattern_from=pattern_from,
				pattern_to=pattern_to
			)
		elif args.method == "databend-aggressive":
			glitch = databend(
				converted_image,
				method="aggressive",
				pattern_type=args.pattern_type,
				pattern_from=pattern_from,
				pattern_to=pattern_to
			)
		else:
			raise TypeError(f"Unknown method \"{args.method}\"")
		
		current_output_path: None
		if args.iterations > 1:
			iteration_number = f"{i+1:03d}" # TODO: Make it dynamic, where if args.iterations len is over 03d, it makes it fit.
			iteration_filename = f"{output_name}-{iteration_number}.{output_format}"
			current_output_path = output_folder / iteration_filename
		else:
			current_output_path = output_path

		reconstruction: bytes | None = reconstruct(
			blob=glitch,
			format=args.format,
			new_format=output_format,
			output_filepath=str(current_output_path),
			primitive=args.primitive
		)

		if args.verbose:
			print(f"Saved: {current_output_path}")
			print(f"Conversion changed data: {image != converted_image}.")
			print(f"Glitch applied: {converted_image != glitch}.")
		
		if args.iterations > 1 and not args.verbose:
			print(f"Completed iteration {i+1}/{args.iterations}")


if __name__ == "__main__":
	main()
