import gradio as gr
from datetime import datetime


# re-render the list when state changes
def show_todos(todos):
    html_content = "<ul>"
    for i, todo in enumerate(todos):
        status = " (done)" if todo["done"] else ""
        html_content += f"<li>{i}. {todo['title']}{status}</li>"
    html_content += "</ul>"
    return html_content


# as far as I know, you can't simply have a component "subscribe" to state changes a la React
def add_todo(title, todos):
    new_todo = {
        "title": title,
        "done": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # not used
    }
    todos.append(new_todo)
    # return both so you update the state and the display
    return todos, show_todos(todos)


def delete_todo(index, todos):
    # check if index is valid
    if 0 <= index < len(todos):
        todos.pop(index)
    else:
        # must raise the error component because it doesn't rely on the queue
        raise gr.Error(f"Index {index} out of bounds")
    return todos, show_todos(todos)


def toggle_todo(index, todos):
    if 0 <= index < len(todos):
        todos[index]["done"] = not todos[index]["done"]
    else:
        raise gr.Error(f"Index {index} out of bounds")
    return todos, show_todos(todos)


markdown = """
# Gradio Todos\n
Demonstrating session state and custom CSS with <span class="gradio"><strong>Gradio</strong></span> 🎨
"""

# use CSS variables so you don't break themes
css = """
.todos:not(.prose) {
    background: var(--block-background-fill) !important;
    box-shadow: var(--block-shadow) !important;
    border: var(--block-border-width) solid var(--border-color-primary) !important;
    border-radius: var(--block-radius) !important;
    padding: var(--block-padding) !important; }
.todos.prose {
    background: var(--input-background-fill) !important;
    box-shadow: var(--input-shadow) !important;
    border: var(--input-border-width) solid var(--input-border-color) !important;
    border-radius: var(--input-radius) !important;
    padding: var(--input-padding) !important; }
.todos, .todos > div {
    height: 100% !important; }
.gradio {
    background: linear-gradient(90deg, #F97700, #F9D100);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; }
footer {
    display: none !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    state = gr.State(value=[])

    gr.Markdown(markdown)

    # todos will be on top on mobile
    with gr.Row(equal_height=True):
        with gr.Column():
            todos_display = gr.HTML(elem_classes="todos")

        with gr.Column():
            todo_input = gr.Textbox(show_label=False)
            add_button = gr.Button("Submit", variant="primary")
            with gr.Row():
                toggle_index_input = gr.Number(minimum=0, precision=0, show_label=False, scale=1)
                toggle_button = gr.Button("✅ Toggle", scale=2)
            with gr.Row():
                delete_index_input = gr.Number(minimum=0, precision=0, show_label=False, scale=1)
                delete_button = gr.Button("🗑️ Delete", variant="stop", scale=2)

    add_button.click(
        fn=add_todo,
        inputs=[todo_input, state],
        outputs=[state, todos_display],
    )
    delete_button.click(
        fn=delete_todo,
        inputs=[delete_index_input, state],
        outputs=[state, todos_display],
    )
    toggle_button.click(
        fn=toggle_todo,
        inputs=[toggle_index_input, state],
        outputs=[state, todos_display],
    )

    # example cache is off by default
    gr.Examples(examples=[["Buy groceries"], ["Read a book"]], inputs=[todo_input])

if __name__ == "__main__":
    demo.launch(
        debug=True,
        share=False,
        show_api=False,
        server_port=7860,
        server_name="0.0.0.0",
    )
