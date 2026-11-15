# Adding an industry profile

Create a JSON file in this directory. Start from `base.json`.

Required keys:
- `profile_id`: lowercase stable identifier.
- `profile_version`: semantic version.
- `description`: what the profile covers.
- `entity_extensions`: profile-specific descriptions for base entity types or new entity types.
- `relations`: typed relation definitions with `name`, `head`, `tail`, and `description`.
- Optional `business_lines` and `event_types`.

Do not create incompatible meanings for base labels. Keep roles mutually exclusive:
- `organization` = named actor or legal/operating entity.
- `person` = named natural person.
- `service` = activity performed or delivered.
- `asset` = object/site/system acted upon.
- `customer_segment` = buyer or beneficiary group.

Profiles are inference schemas, not evidence that a checkpoint has learned a new domain. Domain-specific training still requires reviewed annotations.
