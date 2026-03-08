from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    # Se autodefine como la carpeta donde esté este archivo
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[4])

# Al usarlo:
mi_config = Config()
print(f"La raíz del proyecto es: {mi_config.project_root}")