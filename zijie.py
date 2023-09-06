class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def pathSum( root, targetSum: int):
        res=[]
        def dfs(root,target,cur): 
            if not root:
                return
            now=[]
            if target-root.val>0:
                now.append(root.val)
                dfs(root.left,target-root.val,cur+now)
                dfs(root.right,target-root.val,cur+now)
            elif target-root.val==0 and root.left==None and root.right==None:
                now.append(root.val)
                res.append(cur+now)
            else:
                return
        cur=[]
        dfs(root,targetSum,cur)
        return res        



b=TreeNode(2)
c=TreeNode(3)
# pathSum(b,-5)

a=TreeNode(1,b,c)
def sumNumbers( root) -> int:

        res=[]


        def dfs(root,cur):
            if not root:
                res.append(cur)
                return 

            now=cur*10+root.val
            dfs(root.left,now)
            dfs(root.right,now)
            
        dfs(root,0)
        a=0
        for i in range(len(res)):
            a=a+res[i]
        return a

# sumNumbers(a)

def kthSmallest( root, k: int) -> int:

        res=[]
        def dfs(root):

            if not root:
                return
            if len(res)==k:
                return
            if root.left==None:
                res.append(root.val)
            dfs(root.left)
            dfs(root.right)
        dfs(root)
        return res[k-1]

a=TreeNode(1)
b=TreeNode(2,a)
c=TreeNode(4)
d=TreeNode(3,b,c)
e=TreeNode(6)
f=TreeNode(5,d,e)

kthSmallest(f,3)