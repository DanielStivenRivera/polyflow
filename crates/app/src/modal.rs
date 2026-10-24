//! Native dialogs (message boxes, file pickers) that never block the event
//! loop.
//!
//! A dialog shown from inside an event handler runs its own modal message
//! loop. On Windows that loop feeds the main window's messages back into
//! winit while our handler is still running; winit cannot deliver events then
//! and answers every `WM_PAINT` by requesting another one, so if a redraw was
//! pending the thread spins on paint messages and the dialog takes seconds to
//! appear, or never does. Dialogs therefore run on their own thread, parented
//! to the main window (which the system disables meanwhile), and their answer
//! comes back to the event loop as a [`UserEvent::DialogAnswer`]; what to do
//! with it is a continuation kept on the main thread.

use std::any::Any;
use std::sync::Arc;

use winit::event::{ElementState, WindowEvent};
use winit::window::Window;

use crate::{AppState, UserEvent};

/// A dialog's answer, on its way back to the event loop.
pub type Answer = Box<dyn Any + Send>;

type Continuation = Box<dyn FnOnce(&mut AppState, Answer)>;

/// The dialog being shown, if any.
#[derive(Default)]
pub struct Dialogs {
    pending: Option<Continuation>,
}

impl Dialogs {
    pub fn is_open(&self) -> bool {
        self.pending.is_some()
    }
}

/// Input the window ignores while a dialog is open. The system does not
/// disable the window for us, as the dialog lives on another thread. Releases
/// and modifier changes still go through, so nothing stays stuck pressed.
pub fn is_blocked_input(event: &WindowEvent) -> bool {
    match event {
        WindowEvent::KeyboardInput { event, .. } => event.state == ElementState::Pressed,
        WindowEvent::MouseInput { state, .. } => *state == ElementState::Pressed,
        WindowEvent::MouseWheel { .. }
        | WindowEvent::Touch(_)
        | WindowEvent::PinchGesture { .. }
        | WindowEvent::PanGesture { .. }
        | WindowEvent::RotationGesture { .. }
        | WindowEvent::DoubleTapGesture { .. }
        | WindowEvent::Ime(_)
        | WindowEvent::DroppedFile(_)
        | WindowEvent::HoveredFile(_)
        | WindowEvent::CloseRequested => true,
        _ => false,
    }
}

impl std::fmt::Debug for Dialogs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dialogs")
            .field("open", &self.pending.is_some())
            .finish()
    }
}

impl AppState {
    /// Shows a dialog on its own thread: `dialog` builds an `rfd` dialog
    /// parented to the window it is given and shows it. `then` runs on the
    /// event loop thread with the answer. While a dialog is open, further
    /// requests are ignored.
    pub(crate) fn ask<T: Send + 'static>(
        &mut self,
        dialog: impl FnOnce(&Window) -> T + Send + 'static,
        then: impl FnOnce(&mut AppState, T) + 'static,
    ) {
        if self.dialogs.pending.is_some() {
            return;
        }
        let window = Arc::clone(&self.window);
        let proxy = self.proxy.clone();
        let spawned = std::thread::Builder::new()
            .name("dialog".into())
            .spawn(move || {
                let shown =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dialog(&window)));
                let answer: Answer = match shown {
                    Ok(answer) => Box::new(answer),
                    // Still answer, so the app does not wait forever.
                    Err(_) => Box::new(()),
                };
                // The event loop may exit as soon as it has the answer.
                drop(window);
                let _ = proxy.send_event(UserEvent::DialogAnswer(answer));
            });
        match spawned {
            Ok(_) => {
                self.dialogs.pending = Some(Box::new(move |state, answer| {
                    if let Ok(answer) = answer.downcast::<T>() {
                        then(state, *answer);
                    }
                }));
            }
            Err(e) => self.ui.error = Some(format!("Could not open a dialog: {e}")),
        }
    }

    /// Hands a dialog's answer to whatever asked for it.
    pub(crate) fn dialog_answered(&mut self, answer: Answer) {
        if let Some(then) = self.dialogs.pending.take() {
            then(self, answer);
        }
        self.window.request_redraw();
    }
}
