import tkinter as tk
from tkinter import ttk, messagebox

class UserManagementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("用户管理界面")
        self.root.geometry("400x300")
        self.root.configure(bg='#e0f7fa')

        # 用户在线状态显示框
        self.status_label = ttk.Label(root, text="在线用户:", font=('Arial', 12), background='#e0f7fa')
        self.status_label.pack(pady=10)

        self.online_users_text = tk.Text(root, height=5, width=40, state='disabled')
        self.online_users_text.pack(pady=10)

        # 示例：默认显示一个用户在线
        self.add_user_to_online_list("User1")

        # 添加其他用户管理界面的组件和逻辑...

    def add_user_to_online_list(self, username):
        self.online_users_text.config(state='normal')
        self.online_users_text.insert('end', f"{username} - 在线\n")
        self.online_users_text.config(state='disabled')


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("心电图数据输入")
        self.root.geometry("400x200")
        self.root.configure(bg='#e0f7fa')

        # 使用ttk风格
        style = ttk.Style()
        style.configure("TLabel", background='#e0f7fa', font=('Arial', 12))
        style.configure("TEntry", font=('Arial', 12))
        style.configure("TButton", background='#4caf50', foreground='white', font=('Arial', 12))

        # 账号输入
        self.username_label = ttk.Label(root, text="账号:")
        self.username_label.pack(pady=5)

        self.username_entry = ttk.Entry(root, width=30)
        self.username_entry.pack(pady=5)

        # 密码输入
        self.password_label = ttk.Label(root, text="密码:")
        self.password_label.pack(pady=5)

        self.password_entry = ttk.Entry(root, show="*", width=30)
        self.password_entry.pack(pady=5)

        # 登录按钮
        self.login_button = ttk.Button(root, text="登录", command=self.login)
        self.login_button.pack(pady=10)

        # 忘记密码链接
        self.forgot_password_link = tk.Label(root, text="忘记密码？", fg="blue", cursor="hand2")
        self.forgot_password_link.pack(pady=5)
        self.forgot_password_link.bind("<Button-1>", self.forgot_password)

    def login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()

        # 这里只是一个简单的示例，实际上你应该进行用户认证和授权的逻辑
        if username == "root" and password == "1":
            self.open_user_management()
        else:
            messagebox.showerror("登录失败", "账号或密码错误，请重试。")

    def forgot_password(self, event):
        messagebox.showinfo("忘记密码", "请联系管理员进行密码重置。")

    def open_user_management(self):
        self.root.withdraw()  # 隐藏登录窗口
        user_management_window = tk.Toplevel(self.root)
        user_management_app = UserManagementApp(user_management_window)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()



# # 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # 加载模型
# model = torch.load(r'D:\dk\py37\venv1\代码转接\model\MCA-net\PTB\Best_Model_KFold1.pt', map_location=device)
# model.eval()  # 设置模型为评估模式

# # 预处理数据
# preprocessed_data = preprocess_data(data)

# # 将输入数据转换为PyTorch的Tensor
# input_data = torch.tensor(preprocessed_data, dtype=torch.float).to(device)

# with torch.no_grad():
#     # 将输入数据转换为PyTorch的Tensor
#     input_data = torch.tensor(preprocessed_data, dtype=torch.float).to(device)
#
#     # 进行预测
#     output = model(input_data)

# # 对输出应用Softmax，并获取最大概率的类别
# probabilities = torch.nn.functional.softmax(output, dim=1)
# predicted_class = torch.argmax(probabilities, dim=1).item()

# # 输出最终的预测结果
# print("Predicted Class:", predicted_class)
