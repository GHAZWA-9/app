# Introduction to HTML

---

## What is HTML?

HTML (HyperText Markup Language) is the standard language used to create web pages. It structures content on the web and describes the structure of a webpage. HTML consists of a series of elements that tell the browser how to display the content.

## Key Concepts

### 1. HTML Elements
- An HTML element is defined by a start tag, content, and an end tag.
- Example:
    ```html
    <p>This is a paragraph.</p>
    ```
- In this example, `<p>` is the start tag, and `</p>` is the end tag. The content is "This is a paragraph."

### 2. HTML Tags
- Tags are used to mark the beginning and end of an HTML element. They are enclosed in angle brackets (`< >`).
- Example: `<h1>`, `<p>`, `<div>`, `<a>`

### 3. HTML Attributes
- Attributes provide additional information about an element and are always included in the opening tag.
- Example:
    ```html
    <a href="https://www.example.com">Visit Example</a>
    ```
- In this example, `href` is an attribute of the `<a>` tag, specifying the URL of the link.

### 4. HTML Document Structure
- A basic HTML document has a structure that includes a `<!DOCTYPE html>` declaration, followed by an `<html>` tag that contains two main sections: `<head>` and `<body>`.
- Example:
    ```html
    <!DOCTYPE html>
    <html>
    <head>
      <title>My First Webpage</title>
    </head>
    <body>
      <h1>Welcome to My Website</h1>
      <p>This is my first webpage.</p>
    </body>
    </html>
    ```
- **Explanation:**
  - `<!DOCTYPE html>`: Declares the document type and version of HTML.
  - `<html>`: The root element of an HTML page.
  - `<head>`: Contains metadata about the HTML document (e.g., title, links to stylesheets).
  - `<body>`: Contains the content of the webpage that will be displayed to users.

### 5. Common HTML Tags

- **Headings:** Used to define headings, ranging from `<h1>` (most important) to `<h6>` (least important).
    ```html
    <h1>This is a main heading</h1>
    <h2>This is a subheading</h2>
    ```
- **Paragraphs:** Used to define blocks of text.
    ```html
    <p>This is a paragraph of text.</p>
    ```
- **Links:** Used to create hyperlinks.
    ```html
    <a href="https://www.example.com">Click here</a>
    ```
- **Images:** Used to embed images in a webpage.
    ```html
    <img src="image.jpg" alt="Description of image">
    ```
- **Lists:** Used to create ordered (`<ol>`) or unordered (`<ul>`) lists.
    ```html
    <ul>
      <li>Item 1</li>
      <li>Item 2</li>
    </ul>
    ```

### 6. HTML Comments
- Comments in HTML are not displayed in the browser and are used to leave notes in the code.
- Example:
    ```html
    <!-- This is a comment -->
    ```

### 7. Basic HTML Page Example

Here's a simple example of a complete HTML page:
```html
<!DOCTYPE html>
<html>
<head>
  <title>My First HTML Page</title>
</head>
<body>
  <h1>Hello, World!</h1>
  <p>This is a paragraph on my first HTML page.</p>
  <a href="https://www.example.com">Visit Example Website</a>
</body>
</html>

![alt text](Display.png "Code Execution")

