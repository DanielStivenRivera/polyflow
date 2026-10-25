//! Undo and redo for the retopology mesh.
//!
//! All edits go through [`MeshDocument::edit`], which runs them as atomic
//! transactions: the mesh journals every write while the edit runs; if the
//! edit fails (or panics) the writes are rolled back and the mesh is left
//! exactly as it was, otherwise they become one undo step.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use crate::halfedge::{Change, HalfEdgeMesh};

/// Default number of undo steps kept.
pub const DEFAULT_MAX_STEPS: usize = 1000;

#[derive(Debug)]
struct Transaction {
    name: String,
    merge_key: Option<u64>,
    changes: Vec<Change>,
}

/// A mesh with its undo history.
#[derive(Debug)]
pub struct MeshDocument {
    mesh: HalfEdgeMesh,
    undo: Vec<Transaction>,
    redo: Vec<Transaction>,
    max_steps: usize,
}

impl Default for MeshDocument {
    fn default() -> Self {
        Self::new(HalfEdgeMesh::new())
    }
}

impl MeshDocument {
    pub fn new(mesh: HalfEdgeMesh) -> Self {
        Self {
            mesh,
            undo: Vec::new(),
            redo: Vec::new(),
            max_steps: DEFAULT_MAX_STEPS,
        }
    }

    pub fn mesh(&self) -> &HalfEdgeMesh {
        &self.mesh
    }

    /// Sets how many undo steps are kept, dropping the oldest beyond it.
    pub fn set_max_steps(&mut self, max_steps: usize) {
        self.max_steps = max_steps.max(1);
        self.trim();
    }

    /// Runs `edit` as one undoable step named `name`. If it returns an error
    /// or panics, every change it made is rolled back. Edits that change
    /// nothing leave no undo step.
    pub fn edit<R, E>(
        &mut self,
        name: &str,
        edit: impl FnOnce(&mut HalfEdgeMesh) -> Result<R, E>,
    ) -> Result<R, E> {
        self.run(name, None, edit)
    }

    /// Like [`MeshDocument::edit`], but merges into the previous step if it was
    /// made with the same `merge_key` and nothing was undone since. Use one key
    /// per gesture (e.g. per vertex drag) so a whole gesture undoes at once.
    pub fn edit_merged<R, E>(
        &mut self,
        name: &str,
        merge_key: u64,
        edit: impl FnOnce(&mut HalfEdgeMesh) -> Result<R, E>,
    ) -> Result<R, E> {
        self.run(name, Some(merge_key), edit)
    }

    fn run<R, E>(
        &mut self,
        name: &str,
        merge_key: Option<u64>,
        edit: impl FnOnce(&mut HalfEdgeMesh) -> Result<R, E>,
    ) -> Result<R, E> {
        self.mesh.begin_journal();
        let result = catch_unwind(AssertUnwindSafe(|| edit(&mut self.mesh)));
        let changes = self.mesh.end_journal();
        match result {
            Ok(Ok(value)) => {
                self.commit(name, merge_key, changes);
                Ok(value)
            }
            Ok(Err(e)) => {
                self.mesh.apply_changes(&changes, true);
                Err(e)
            }
            Err(panic) => {
                self.mesh.apply_changes(&changes, true);
                resume_unwind(panic)
            }
        }
    }

    fn commit(&mut self, name: &str, merge_key: Option<u64>, changes: Vec<Change>) {
        if changes.is_empty() {
            return;
        }
        let merge = merge_key.is_some()
            && self.redo.is_empty()
            && self
                .undo
                .last()
                .is_some_and(|top| top.merge_key == merge_key);
        if merge {
            let top = self.undo.last_mut().expect("checked above");
            // The gesture is named for its last part (a drag that ends in a
            // weld is a weld).
            if top.name != name {
                top.name = name.to_owned();
            }
            top.changes.extend(changes);
            top.changes = compact(std::mem::take(&mut top.changes));
            if top.changes.is_empty() {
                // The merged gesture ended where it started.
                self.undo.pop();
            }
            return;
        }
        let changes = compact(changes);
        if changes.is_empty() {
            return;
        }
        self.redo.clear();
        self.undo.push(Transaction {
            name: name.to_owned(),
            merge_key,
            changes,
        });
        self.trim();
    }

    fn trim(&mut self) {
        if self.undo.len() > self.max_steps {
            let excess = self.undo.len() - self.max_steps;
            self.undo.drain(..excess);
        }
    }

    /// Undoes the last step; returns its name.
    pub fn undo(&mut self) -> Option<&str> {
        let transaction = self.undo.pop()?;
        self.mesh.apply_changes(&transaction.changes, true);
        self.redo.push(transaction);
        self.redo.last().map(|t| t.name.as_str())
    }

    /// Redoes the last undone step; returns its name.
    pub fn redo(&mut self) -> Option<&str> {
        let transaction = self.redo.pop()?;
        self.mesh.apply_changes(&transaction.changes, false);
        self.undo.push(transaction);
        self.undo.last().map(|t| t.name.as_str())
    }

    /// Name of the step [`MeshDocument::undo`] would undo.
    pub fn undo_name(&self) -> Option<&str> {
        self.undo.last().map(|t| t.name.as_str())
    }

    /// Name of the step [`MeshDocument::redo`] would redo.
    pub fn redo_name(&self) -> Option<&str> {
        self.redo.last().map(|t| t.name.as_str())
    }

    pub fn undo_steps(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_steps(&self) -> usize {
        self.redo.len()
    }

    /// Forgets all undo and redo steps.
    pub fn clear_history(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

/// Keeps one change per element, with its first old and last new value, and
/// drops elements that ended where they started. Each change stores whole
/// element values, so applying the compacted changes in any order gives the
/// same result as replaying them all.
fn compact(changes: Vec<Change>) -> Vec<Change> {
    #[derive(Hash, PartialEq, Eq)]
    enum Key {
        Vertex(u32),
        HalfEdge(u32),
        Face(u32),
    }
    let mut index: HashMap<Key, usize> = HashMap::with_capacity(changes.len());
    let mut out: Vec<Change> = Vec::with_capacity(changes.len());
    for change in changes {
        let key = match &change {
            Change::Vertex(i, ..) => Key::Vertex(*i),
            Change::HalfEdge(i, ..) => Key::HalfEdge(*i),
            Change::Face(i, ..) => Key::Face(*i),
        };
        match index.get(&key) {
            Some(&at) => match (&mut out[at], change) {
                (Change::Vertex(_, _, new), Change::Vertex(_, _, latest)) => *new = latest,
                (Change::HalfEdge(_, _, new), Change::HalfEdge(_, _, latest)) => *new = latest,
                (Change::Face(_, _, new), Change::Face(_, _, latest)) => *new = latest,
                _ => unreachable!("keys distinguish element kinds"),
            },
            None => {
                index.insert(key, out.len());
                out.push(change);
            }
        }
    }
    out.retain(|c| match c {
        Change::Vertex(_, old, new) => old != new,
        Change::HalfEdge(_, old, new) => old != new,
        Change::Face(_, old, new) => old != new,
    });
    out
}

#[cfg(test)]
mod tests {
    use glam::Vec3;

    use super::*;
    use crate::halfedge::{TopologyError, VertexId};

    fn quad(doc: &mut MeshDocument) -> [VertexId; 4] {
        doc.edit("Add quad", |m| {
            let v =
                [Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y].map(|p| m.add_vertex(p));
            m.add_face(&v)?;
            Ok::<_, TopologyError>(v)
        })
        .unwrap()
    }

    #[test]
    fn undo_and_redo_restore_exact_states() {
        let mut doc = MeshDocument::default();
        let empty = doc.mesh().clone();
        let v = quad(&mut doc);
        let one_quad = doc.mesh().clone();
        doc.edit("Move", |m| {
            m.set_position(v[2], Vec3::new(2.0, 2.0, 0.0));
            Ok::<_, ()>(())
        })
        .unwrap();
        let moved = doc.mesh().clone();

        assert_eq!(doc.undo(), Some("Move"));
        assert_eq!(*doc.mesh(), one_quad);
        assert_eq!(doc.undo(), Some("Add quad"));
        assert_eq!(*doc.mesh(), empty);
        assert_eq!(doc.mesh().vertex_count(), 0);
        assert_eq!(doc.undo(), None);
        assert_eq!(doc.redo(), Some("Add quad"));
        assert_eq!(doc.redo(), Some("Move"));
        assert_eq!(*doc.mesh(), moved);
        doc.mesh().validate().unwrap();
    }

    #[test]
    fn failed_edits_leave_no_trace() {
        let mut doc = MeshDocument::default();
        let v = quad(&mut doc);
        let before = doc.mesh().clone();
        let version = doc.mesh().version();
        // Adds a vertex, then fails: the vertex must disappear too.
        let result = doc.edit("Bad face", |m| {
            m.add_vertex(Vec3::Z);
            m.add_face(&[v[0], v[1], v[2]]) // Same orientation as the quad: refused.
        });
        assert!(matches!(result, Err(TopologyError::EdgeTaken(..))));
        assert_eq!(*doc.mesh(), before);
        assert_eq!(doc.undo_steps(), 1);
        assert_ne!(
            doc.mesh().version(),
            version,
            "views must refresh after a rollback"
        );
        doc.mesh().validate().unwrap();
    }

    #[test]
    fn panicking_edits_are_rolled_back() {
        let mut doc = MeshDocument::default();
        quad(&mut doc);
        let before = doc.mesh().clone();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = doc.edit("Explode", |m| {
                m.add_vertex(Vec3::Z);
                panic!("tool bug");
                #[allow(unreachable_code)]
                Ok::<_, ()>(())
            });
        }));
        assert!(result.is_err());
        assert_eq!(*doc.mesh(), before);
        // The document is still usable.
        quad(&mut doc);
        assert_eq!(doc.mesh().face_count(), 2);
    }

    #[test]
    fn merged_edits_undo_as_one_step() {
        let mut doc = MeshDocument::default();
        let v = quad(&mut doc);
        let before_drag = doc.mesh().clone();
        for i in 1..=50 {
            doc.edit_merged("Move vertex", 7, |m| {
                m.set_position(v[0], Vec3::new(i as f32 * 0.01, 0.0, 0.0));
                Ok::<_, ()>(())
            })
            .unwrap();
        }
        assert_eq!(doc.undo_steps(), 2);
        assert_eq!(doc.undo(), Some("Move vertex"));
        assert_eq!(*doc.mesh(), before_drag);

        // A gesture that returns to the start leaves no step behind.
        doc.redo();
        let original = doc.mesh().position(v[0]);
        for x in [0.5, 0.25, original.x] {
            doc.edit_merged("Move vertex", 8, |m| {
                m.set_position(v[0], Vec3::new(x, original.y, original.z));
                Ok::<_, ()>(())
            })
            .unwrap();
        }
        assert_eq!(doc.undo_steps(), 2);
    }

    #[test]
    fn new_edits_clear_redo_and_history_is_bounded() {
        let mut doc = MeshDocument::default();
        quad(&mut doc);
        quad(&mut doc);
        doc.undo();
        assert_eq!(doc.redo_steps(), 1);
        quad(&mut doc);
        assert_eq!(doc.redo_steps(), 0);

        doc.set_max_steps(3);
        for _ in 0..5 {
            quad(&mut doc);
        }
        assert_eq!(doc.undo_steps(), 3);
    }

    #[test]
    fn undo_restores_the_same_ids() {
        let mut doc = MeshDocument::default();
        let v = quad(&mut doc);
        let face = doc.mesh().faces().next().unwrap();
        doc.edit("Delete", |m| m.remove_vertex(v[1])).unwrap();
        assert!(!doc.mesh().contains_face(face));
        doc.undo();
        assert!(doc.mesh().contains_face(face));
        assert_eq!(
            doc.mesh().face_vertices(face).collect::<Vec<_>>(),
            v.to_vec()
        );
        // Allocation after undo still finds free slots correctly.
        quad(&mut doc);
        doc.mesh().validate().unwrap();
    }
}
