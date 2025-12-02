***

## YAML (.yml) Configuration Guide

### Basic Syntax Rules

- Indentation expresses structure. **Always use spaces, not tabs.**
- Key-value pairs use a colon:

```
key: value
```

- Nested dictionaries use indentation:

```
parent:
  child: value
```

- Lists use a dash (`-`):

```
fruits:
  - Apple
  - Banana
  - Cherry
```

- Strings, numbers, booleans, and null values are supported:

```
name: John Doe
age: 30
is_member: true
score: null
```

- Comments start with `#`:

```
# This is a comment
```


### Advanced Features

- Anchors (`&`) and aliases (`*`) for repeated configuration:

```
base: &default
  retries: 3
  timeout: 30

service1:
  <<: *default
  host: example.com
```

- Multi-line strings using `|` (literal) or `>` (folded):

```
description: |
  This is a
  multi-line string.
note: >
  This will
  be one single line.
```


### File Extensions

- Use `.yml` or `.yaml`—both are equivalent for YAML files.[^3_8]


### Example YAML Config

```yaml
model:
  type: resnet
  layers: 34
  pretrained: true

optimizer:
  lr: 0.01
  weight_decay: 0.0001

users:
  - name: Alice
    admin: true
  - name: Bob
    admin: false
```


***

## Loading a YAML Config in Python

Use the `pyyaml` library for reading YAML files:

```python
import yaml

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
# config is now a Python dictionary
```

- `safe_load()` is recommended for security.
- Access nested values with dictionary keys:

```python
lr = config["optimizer"]["lr"]
model_type = config["model"]["type"]
```


***
