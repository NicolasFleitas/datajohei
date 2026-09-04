# Cómo contribuir

Gracias por querer mejorar DataJohei. El camino habitual es: fork → rama → PR.

## Camino rápido

1. Hacé un fork del repositorio.
2. Cloná tu fork y creá una rama descriptiva desde `main`.
3. Hacé tus cambios con commits claros.
4. Pusheá tu rama y abrí un PR hacia `main`.

```bash
git clone https://github.com/TU_USUARIO/datajohei.git && cd datajohei
git checkout main && git pull origin main
git checkout -b feat/descripcion-corta
# ... hacé tus cambios ...
git push -u origin feat/descripcion-corta
```

## Reglas para ramas y commits

- Ramá desde `main` con nombres descriptivos: `feat/...`, `fix/...` o `docs/...`.
- Un cambio por PR. Mantené el diff chico y enfocado.
- Escribí commits claros en español o inglés, por ejemplo: `Agrega validación de CSV vacío`.

## Checklist antes del PR

- [ ] La rama parte de `main` actualizado.
- [ ] Describiste qué cambia y cómo probarlo.
- [ ] Si toca lógica de `src/`, agregaste o actualizaste tests (`uv run pytest tests/ -q`).

## Cómo reportar issues

Abrí un issue con:

1. Qué esperabas y qué pasó en realidad.
2. Pasos para reproducirlo.
3. Versión de Python, cómo lo ejecutás (local o Docker) y archivo de ejemplo si aplica.
4. Captura o mensaje de error si hay uno.
