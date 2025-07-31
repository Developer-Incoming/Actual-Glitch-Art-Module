# *A*ctual *G*litch *A*rt *M*odule

Produces authentic glitched images with a wide varity of configurations for you to play with.

---
### Original:
![Original](https://github.com/Developer-Incoming/Actual-Glitch-Art-Module/blob/main/Examples/Single_lavender_flower02.jpg?raw=true)
---
### Example Glitch 1
```bash
agam.py Single_lavender_flower02.jpg Glitched_lavender.png --format jpg
```
![Glitched1](https://github.com/Developer-Incoming/Actual-Glitch-Art-Module/blob/main/Examples/Glitched_lavender.png?raw=true)
---
### Example Glitch 2
```bash
agam.py Single_lavender_flower02.jpg Glitched_lavender.png --format jpg --method databend-aggressive --pattern-type bit_shift_xor --pattern-from 56 --pattern-to 65
```
![Glitched2](https://github.com/Developer-Incoming/Actual-Glitch-Art-Module/blob/main/Examples/Glitched_lavender2.png?raw=true)
---

Disclaimer: The code is messy and may include a lot of debugging code residing somewhere in.
